"""Does the sim mug (libero_90 #72) pick up from every point on its rim? Scripted radial rim pinch at 12 angles.

Rule (written before running): if the failures cluster in one arc that contains the take05 grasp angle (84 deg)
and the take04 angle (40 deg) succeeds, the handle explains take05; otherwise look elsewhere.
"""

import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

from handsim.sim import GRIP_CLOSE, GRIP_OPEN, TASKS, eef_pose, follow, make_env, obj_pose

TASK = TASKS["mug"]
DEGREES = [int(a) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(0, 360, 30))


def plan(obs, angle: float, lift: float = 0.15):
    pick, _ = obj_pose(obs, TASK.pick)
    place, _ = obj_pose(obs, TASK.place)
    _, q0 = eef_pose(obs)
    down = R.from_quat(q0).as_matrix()
    u = np.array([np.cos(angle), np.sin(angle), 0])
    yaw = angle - np.arctan2(down[1, 1], down[0, 1])  # closing axis (eef y) radial
    q = R.from_matrix(R.from_euler("z", yaw).as_matrix() @ down).as_quat()
    dz = TASK.grasp_tip_z + 0.011
    g = pick + TASK.rim_radius * u + [0, 0, dz]
    p = place + TASK.rim_radius * u + [0, 0, dz + 0.02]
    return [
        (g + [0, 0, lift], q, GRIP_OPEN, 0), (g, q, GRIP_OPEN, 0), (g, q, GRIP_CLOSE, 12),
        (g + [0, 0, lift], q, GRIP_CLOSE, 0), (p + [0, 0, lift], q, GRIP_CLOSE, 0), (p, q, GRIP_CLOSE, 0),
        (p, q, GRIP_OPEN, 12), (p + [0, 0, lift], q, GRIP_OPEN, 0),
    ]


def main(out_dir: str) -> None:
    res = []
    for deg in DEGREES:
        env, _, obs = make_env(task_id=TASK.task_id)
        _, ok, _ = follow(env, obs, plan(obs, np.radians(deg)))
        res.append({"angle_deg": deg, "success": ok})
        print(res[-1], flush=True)
        env.close()
    with open(f"{out_dir}/rim_angle_sweep_{DEGREES[0]}_{DEGREES[-1]}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
