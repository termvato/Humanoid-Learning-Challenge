"""Scripted pick-and-place with ground-truth object poses: tests the controller, grasp height and success check."""

import json
import sys

import imageio

from handsim.sim import GRIP_CLOSE, GRIP_OPEN, eef_pose, follow, make_env, obj_pose


def plan(obs, grasp_dz: float, rim_dy: float, lift: float = 0.12):
    bowl, _ = obj_pose(obs, "akita_black_bowl_1")
    plate, _ = obj_pose(obs, "plate_1")
    _, q = eef_pose(obs)  # keep the start orientation (pointing down)
    g = bowl + [0, rim_dy, grasp_dz]
    p = plate + [0, rim_dy, grasp_dz + 0.02]
    return [
        (g + [0, 0, lift], q, GRIP_OPEN, 0),
        (g, q, GRIP_OPEN, 0),
        (g, q, GRIP_CLOSE, 10),
        (g + [0, 0, lift], q, GRIP_CLOSE, 0),
        (p + [0, 0, lift], q, GRIP_CLOSE, 0),
        (p, q, GRIP_CLOSE, 0),
        (p, q, GRIP_OPEN, 10),
        (p + [0, 0, lift], q, GRIP_OPEN, 0),
    ]


def main(out_dir: str) -> None:
    results = []
    for grasp_dz in (0.03, 0.04, 0.05):
        for rim_dy in (-0.05, -0.045, 0.045, 0.05):
            env, _, obs = make_env()
            frames = []
            obs, ok, steps = follow(env, obs, plan(obs, grasp_dz, rim_dy), frames=frames)
            results.append({"grasp_dz": grasp_dz, "rim_dy": rim_dy, "success": ok, "steps": steps})
            print(results[-1], flush=True)
            if ok and not any(r["success"] for r in results[:-1]):
                imageio.mimsave(f"{out_dir}/oracle.mp4", frames, fps=20)
            env.close()
    with open(f"{out_dir}/result.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
