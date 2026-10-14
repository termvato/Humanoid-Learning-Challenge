"""Scripted pick-and-place for libero_90 #72 (white mug on the plate) with true poses: grasp sweep + robustness."""

import json
import sys

import imageio
import numpy as np

from handsim.sim import GRIP_CLOSE, GRIP_OPEN, eef_pose, follow, make_env, obj_pose

TASK = 72
PICK, PLACE = "porcelain_mug_1", "plate_1"


def plan(obs, grasp_dz: float, rim_dy: float, lift: float = 0.15):
    pick, _ = obj_pose(obs, PICK)
    place, _ = obj_pose(obs, PLACE)
    _, q = eef_pose(obs)
    g = pick + [0, rim_dy, grasp_dz]
    p = place + [0, rim_dy, grasp_dz + 0.02]
    return [
        (g + [0, 0, lift], q, GRIP_OPEN, 0), (g, q, GRIP_OPEN, 0), (g, q, GRIP_CLOSE, 12),
        (g + [0, 0, lift], q, GRIP_CLOSE, 0), (p + [0, 0, lift], q, GRIP_CLOSE, 0), (p, q, GRIP_CLOSE, 0),
        (p, q, GRIP_OPEN, 12), (p + [0, 0, lift], q, GRIP_OPEN, 0),
    ]


def main(out_dir: str) -> None:
    sweep = []
    for dz in (0.09, 0.10, 0.105):
        for dy in (-0.045, -0.04, -0.035, 0.035, 0.04, 0.045):
            env, _, obs = make_env(task_id=TASK)
            frames = []
            _, ok, _ = follow(env, obs, plan(obs, dz, dy), frames=frames)
            sweep.append({"grasp_dz": dz, "rim_dy": dy, "success": ok})
            print(sweep[-1], flush=True)
            if ok and not any(r["success"] for r in sweep[:-1]):
                imageio.mimsave(f"{out_dir}/oracle_mug.mp4", frames, fps=20)
            env.close()
    best = max(sweep, key=lambda r: r["success"])
    stored = []
    if best["success"]:
        for i in range(50):
            env, _, obs = make_env(task_id=TASK, init=i)
            _, ok, _ = follow(env, obs, plan(obs, best["grasp_dz"], best["rim_dy"]))
            stored.append(ok)
            env.close()
    res = {"sweep": sweep, "best": best, "stored_success": float(np.mean(stored)) if stored else None}
    with open(f"{out_dir}/oracle_mug.json", "w") as f:
        json.dump(res, f, indent=1)
    print("best", best, "stored", res["stored_success"])


if __name__ == "__main__":
    main(sys.argv[1])
