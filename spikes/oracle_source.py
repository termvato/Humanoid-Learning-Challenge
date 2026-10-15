"""Scripted source trajectory for the demo multiplier: the oracle rim pinch (true mug pose) written in the same
traj_sim.npz format as a retargeted phone take. Feeding it through the same multiplier and ACT recipe separates
"human-derived data" from "learner" as the cause of the policy's success rate.

Usage: MUJOCO_GL=egl .venv/bin/python spikes/oracle_source.py outputs/oracle/traj_sim.npz
"""

import sys
from pathlib import Path

import numpy as np

from handsim.sim import GRIP_CLOSE, GRIP_OPEN, TASKS, eef_pose, make_env, obj_pose

TASK = TASKS["mug"]
STEP = 0.01  # m per control step while moving (about the human takes' speed)
DWELL = 12  # steps held while the gripper closes or opens, as in tools/replay.py


def segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / STEP)))
    return a + (b - a) * np.linspace(0, 1, n)[1:, None]


def main(out: str) -> None:
    env, _, obs = make_env(task_id=TASK.task_id)
    pick, _ = obj_pose(obs, TASK.pick)
    place, _ = obj_pose(obs, TASK.place)
    p0, q0 = eef_pose(obs)
    dz, dy, lift = TASK.grasp_tip_z + 0.011, -0.04, 0.15  # oracle sweep: rim along -y, eef 9.5 cm above the base
    g = pick + [0, dy, dz]
    r = place + [0, dy, dz + 0.02]
    legs = [  # (target, grip, dwell after)
        (g + [0, 0, lift], GRIP_OPEN, 0),
        (g, GRIP_OPEN, 0),
        (g, GRIP_CLOSE, DWELL),
        (g + [0, 0, lift], GRIP_CLOSE, 0),
        (r + [0, 0, lift], GRIP_CLOSE, 0),
        (r, GRIP_CLOSE, 0),
        (r, GRIP_OPEN, DWELL),
        (r + [0, 0, lift], GRIP_OPEN, 0),
    ]
    eef, grip, cur = [], [], p0
    for tgt, gr, dwell in legs:
        seg = segment(cur, tgt) if np.linalg.norm(tgt - cur) > 1e-6 else tgt[None]
        eef += list(seg) + [tgt] * dwell
        grip += [gr] * (len(seg) + dwell)
        cur = tgt
    eef = np.array(eef)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=np.arange(len(eef)) / 20, eef=eef, quat=np.tile(q0, (len(eef), 1)), grip=np.array(grip),
             pick=pick, place=place, task="mug")
    print(out, len(eef), "steps")


if __name__ == "__main__":
    main(sys.argv[1])
