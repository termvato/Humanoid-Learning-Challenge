"""Hand track + scene -> Panda end-effector path in LIBERO -> open-loop replay (run with .venv).

The sim scene copies the real layout: the sim plate is moved so that it sits where the real plate sat relative
to the real bowl (bowl -> bowl, bowl-to-plate direction matched, scale 1).

Usage: .venv/bin/python tools/replay.py data/raw/take01 outputs/take01 [--orient yaw|full|fixed]
Writes: replay.mp4 (sim), side_by_side.mp4 (phone | sim), traj_sim.npz, replay.json
"""

import argparse
import json
from pathlib import Path

import cv2
import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

from handsim.geometry import PANDA_TIP_OFFSET, fill_gaps, grip_closed, gripper_pose, real_to_sim, smooth_track
from handsim.io.stray import load_stray
from handsim.sim import (
    GRIP_CLOSE,
    GRIP_OPEN,
    eef_pose,
    follow,
    make_env,
    obj_pose,
    pose_to_action,
    set_obj_pose,
)

SIM_TABLE_Z = 0.90
CONTROL_HZ = 20
GRIP_DWELL = 12  # control steps held still at each open/close switch: the Panda gripper needs ~0.5 s, a hand ~0.2 s


def hand_track(hand: dict, T_tw: np.ndarray):
    """Table-frame gripper track: (t, pos, rot, width, valid) from the world-frame keypoints."""
    ok = hand["found"] & hand["depth_ok"]
    n = len(ok)
    pos, rot, width = np.zeros((n, 3)), np.tile(np.eye(3), (n, 1, 1)), np.zeros(n)
    for i in np.flatnonzero(ok):
        kp = hand["kp_world"][i] @ T_tw[:3, :3].T + T_tw[:3, 3]
        pos[i], rot[i], width[i] = gripper_pose(kp)
    flat = np.c_[pos, rot.reshape(n, 9), width]
    flat, valid = fill_gaps(flat, ok)
    return hand["t"], flat[:, :3], flat[:, 3:12].reshape(n, 3, 3), flat[:, 12], valid


def longest_run(mask: np.ndarray) -> slice:
    best, start, cur = (0, 0), None, 0
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start > cur:
                best, cur = (start, i), i - start
            start = None
    return slice(*best)


def symmetric_closest(rot: np.ndarray, start: np.ndarray) -> np.ndarray:
    """The gripper looks the same turned 180 degrees about its approach axis: per frame, pick the version
    closer to the previous one (the first to `start`), so the wrist never spins half a turn for nothing."""
    flip = np.diag([-1.0, -1.0, 1.0])
    out, prev = rot.copy(), start
    for t, Rm in enumerate(rot):
        alt = Rm @ flip
        if np.trace(prev.T @ alt) > np.trace(prev.T @ Rm):
            out[t] = alt
        prev = out[t]
    return out


def add_dwell(eef: np.ndarray, quat: np.ndarray, grip: np.ndarray, n: int):
    """Repeat the frame at each grip switch n times so the robot holds still while its fingers move."""
    idx = []
    for i in range(len(grip)):
        idx.append(i)
        if i > 0 and grip[i] != grip[i - 1]:
            idx += [i] * n
    idx = np.array(idx)
    return eef[idx], quat[idx], grip[idx], idx


def yaw_only(rot_sim: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Keep the start (pointing down) orientation, turned about z to follow the hand's closing axis.
    The gripper is symmetric, so the yaw is taken modulo 180 degrees, closest to the previous one."""
    out, prev = np.zeros_like(rot_sim), 0.0
    base_y = down[:, 1]
    for t, Rm in enumerate(rot_sim):
        y = Rm[:, 1].copy()
        y[2] = 0
        a = np.arctan2(y[1], y[0]) - np.arctan2(base_y[1], base_y[0])
        a = prev + (a - prev + np.pi / 2) % np.pi - np.pi / 2
        out[t], prev = R.from_euler("z", a).as_matrix() @ down, a
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--orient", choices=["yaw", "full", "fixed"], default="yaw")
    args = ap.parse_args()
    out = Path(args.out)
    scene = json.loads((out / "scene.json").read_text())
    hand = dict(np.load(out / "hand.npz"))
    T_tw = np.linalg.inv(np.array(scene["T_world_table"]))

    t, pos, rot, width, valid = hand_track(hand, T_tw)
    run = longest_run(valid)
    t, pos, rot, width = t[run], pos[run], rot[run], width[run]
    fps = 1 / np.median(np.diff(t))
    pos, rot, width = smooth_track(pos, rot, width, fps)
    closed, thr = grip_closed(width)

    env, _, obs = make_env()
    bowl_sim, _ = obj_pose(obs, "akita_black_bowl_1")
    _, pq = obj_pose(obs, "plate_1")
    bowl_r, plate_r = np.array(scene["bowl"]["centre"]), np.array(scene["plate"]["centre"])
    sim_dir = np.array([0.0, 1.0])  # LIBERO: plate sits along +y from the bowl
    plate_sim_xy = bowl_sim[:2] + sim_dir * np.linalg.norm(plate_r - bowl_r)
    obs = set_obj_pose(env, "plate_1", np.r_[plate_sim_xy, SIM_TABLE_Z + 0.005], pq)
    T_st = real_to_sim(bowl_r, plate_r, np.r_[bowl_sim[:2], SIM_TABLE_Z], np.r_[plate_sim_xy, SIM_TABLE_Z])

    pos_s = pos @ T_st[:3, :3].T + T_st[:3, 3]
    rot_s = T_st[:3, :3] @ rot
    _, q0 = eef_pose(obs)
    down = R.from_quat(q0).as_matrix()
    if args.orient == "yaw":
        rot_s = yaw_only(rot_s, down)
    elif args.orient == "fixed":
        rot_s = np.tile(down, (len(rot_s), 1, 1))
    else:
        rot_s = symmetric_closest(rot_s, down)
    eef_s = pos_s - PANDA_TIP_OFFSET * rot_s[:, :, 2]
    quat_s = R.from_matrix(rot_s).as_quat()

    # resample to the control rate
    tc = np.arange(t[0], t[-1], 1 / CONTROL_HZ)
    k = np.clip(np.searchsorted(t, tc), 0, len(t) - 1)
    eef_c, quat_c, grip_c = eef_s[k], quat_s[k], np.where(closed[k], GRIP_CLOSE, GRIP_OPEN)
    eef_c, quat_c, grip_c, src = add_dwell(eef_c, quat_c, grip_c, GRIP_DWELL)
    t_src = tc[src]  # phone time of each control step (repeats during dwells)
    tc = tc[0] + np.arange(len(eef_c)) / CONTROL_HZ  # sim time, longer than the human's by the dwells

    frames = []
    obs, _, _ = follow(env, obs, [(eef_c[0], quat_c[0], GRIP_OPEN, 0)], max_steps_per_target=80, frames=frames)
    n_lead = len(frames)
    success, err = False, []
    for i in range(len(tc)):
        p, q = eef_pose(obs)
        err.append(float(np.linalg.norm(p - eef_c[max(i - 1, 0)])))
        obs, *_ = env.step(pose_to_action(p, q, eef_c[i], quat_c[i], grip_c[i]))
        frames.append(obs["agentview_image"][::-1])
        success = success or bool(env.env._check_success())
    for _ in range(20):
        obs, *_ = env.step(np.r_[np.zeros(6), grip_c[-1]])
        frames.append(obs["agentview_image"][::-1])
        success = success or bool(env.env._check_success())

    # S3: gripper point vs the bowl rim at the first closed frame
    g = int(np.argmax(closed)) if closed.any() else None
    s3 = None
    if g is not None:
        r = np.linalg.norm(pos[g, :2] - bowl_r)
        s3 = {
            "radial_to_rim_m": float(r - scene["bowl"]["diameter"] / 2),
            "height_vs_rim_m": float(pos[g, 2] - scene["bowl"]["height"]),
        }

    imageio.mimsave(out / "replay.mp4", frames, fps=CONTROL_HZ)
    if args.capture != "none":
        side_by_side(args.capture, hand, run, t, t_src, frames[n_lead:], out / "side_by_side.mp4")
    np.savez(out / "traj_sim.npz", t=tc, eef=eef_c, quat=quat_c, grip=grip_c, bowl=bowl_sim, plate=plate_sim_xy)
    res = {
        "success": success,
        "orient": args.orient,
        "track_seconds": float(t[-1] - t[0]),
        "valid_fraction": float(valid.mean()),
        "grip_threshold_m": thr,
        "closed_fraction": float(closed.mean()),
        "tracking_error_median_m": float(np.median(err)),
        "real_bowl_plate_m": float(np.linalg.norm(plate_r - bowl_r)),
        "s3": s3,
    }
    (out / "replay.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def side_by_side(capture: str, hand: dict, run: slice, t: np.ndarray, tc: np.ndarray, sim_frames, path: Path):
    """Phone frame (with the tracked thumb and index tips) next to the sim frame at the same time."""
    cap = load_stray(capture)
    rows = hand["rows"][run]
    want = {int(cap.frames[r]): j for j, r in enumerate(rows)}
    video = cv2.VideoCapture(str(Path(capture) / "rgb.mp4"))
    phone, i = {}, 0
    while len(phone) < len(want):
        ok, img = video.read()
        if not ok:
            break
        if i in want:
            j = want[i]
            kp = hand["kp_2d"][run][j]
            if np.isfinite(kp).all():
                for p in (4, 8):
                    cv2.circle(img, (int(kp[p, 0]), int(kp[p, 1])), 12, (0, 255, 0), -1)
            h = 256
            phone[j] = cv2.cvtColor(cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)), cv2.COLOR_BGR2RGB)
        i += 1
    out = []
    for c, sim in enumerate(sim_frames[: len(tc)]):
        j = min(int(np.searchsorted(t, tc[c])), len(rows) - 1)
        if j in phone:
            out.append(np.concatenate([phone[j], sim], 1))
    if out:
        imageio.mimsave(path, out, fps=CONTROL_HZ)


if __name__ == "__main__":
    main()
