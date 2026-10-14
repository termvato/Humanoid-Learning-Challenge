"""Stray capture -> per-frame right-hand keypoints in the ARKit world frame (run with .venv-hand).

WiLoR gives the hand's shape (21 keypoints, metric, relative) and 2D keypoints. The camera-frame position is
solved from the 2D keypoints with the phone's true intrinsics, then shifted along the viewing ray so that the
keypoints agree with LiDAR depth (median offset). Output: <out>/hand.npz.

Usage: .venv-hand/bin/python tools/extract_hand.py data/raw/take01 outputs/take01 [--stride 2]
"""

import argparse
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch

from handsim.geometry import solve_translation
from handsim.io.stray import load_stray

warnings.filterwarnings("ignore")


def depth_offset(
    kp_cam: np.ndarray, depth: np.ndarray, conf: np.ndarray, K_d: np.ndarray, max_diff: float = 0.06
):
    """Median LiDAR minus predicted depth over keypoints that land on valid, nearby depth. None if < 4 agree."""
    u = np.round(kp_cam[:, 0] / kp_cam[:, 2] * K_d[0, 0] + K_d[0, 2]).astype(int)
    v = np.round(kp_cam[:, 1] / kp_cam[:, 2] * K_d[1, 1] + K_d[1, 2]).astype(int)
    h, w = depth.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = np.full(len(kp_cam), np.nan)
    d[inside] = np.where(conf[v[inside], u[inside]] >= 1, depth[v[inside], u[inside]], np.nan)
    diff = d - kp_cam[:, 2]
    good = np.isfinite(diff) & (np.abs(diff) < max_diff)
    return (float(np.median(diff[good])), int(good.sum())) if good.sum() >= 4 else (None, int(good.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=2, help="use every n-th frame (60 fps video -> 30 Hz)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16, verbose=False)
    cap = load_stray(args.capture)
    video = cv2.VideoCapture(str(Path(args.capture) / "rgb.mp4"))

    n = len(cap)
    rows = range(0, n, args.stride)
    kp_world = np.full((len(rows), 21, 3), np.nan)
    kp_2d = np.full((len(rows), 21, 2), np.nan)
    found = np.zeros(len(rows), bool)
    depth_ok = np.zeros(len(rows), bool)
    offsets = np.full(len(rows), np.nan)
    frame_i = -1
    for r, i in enumerate(rows):
        img = None
        while frame_i < cap.frames[i]:
            ok, img = video.read()
            frame_i += 1
            if not ok:
                break
        if img is None:
            break
        preds = [o for o in pipe.predict(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if o["is_right"] > 0.5]
        if not preds:
            continue
        o = max(preds, key=lambda o: np.prod(np.asarray(o["hand_bbox"])[2:] - np.asarray(o["hand_bbox"])[:2]))
        p = o["wilor_preds"]
        rel, uv = p["pred_keypoints_3d"][0], p["pred_keypoints_2d"][0]
        K = cap.K[i] * np.array([[img.shape[1] / cap.rgb_size[0]], [img.shape[0] / cap.rgb_size[1]], [1]])
        kp_cam = rel + solve_translation(rel, uv, K)
        off, _ = depth_offset(kp_cam, cap.depth(i), cap.confidence(i), cap.K_depth(i))
        if off is not None:
            z = kp_cam[:, 2].mean()
            kp_cam = kp_cam * (z + off) / z  # slide the hand along the viewing ray
            depth_ok[r], offsets[r] = True, off
        T = cap.T_wc[i]
        kp_world[r] = kp_cam @ T[:3, :3].T + T[:3, 3]
        kp_2d[r], found[r] = uv, True
        if r % 50 == 0:
            print(
                f"{r}/{len(rows)} found {found[: r + 1].mean():.2f} depth_ok {depth_ok[: r + 1].mean():.2f}",
                flush=True,
            )

    np.savez(
        out / "hand.npz",
        kp_world=kp_world,
        kp_2d=kp_2d,
        found=found,
        depth_ok=depth_ok,
        depth_offset=offsets,
        rows=np.array(list(rows)),
        t=cap.timestamps[list(rows)],
    )
    print(
        f"found {found.mean():.2f}, depth-corrected {depth_ok.mean():.2f}, "
        f"median |offset| {np.nanmedian(np.abs(offsets)) * 100:.1f} cm"
    )


if __name__ == "__main__":
    main()
