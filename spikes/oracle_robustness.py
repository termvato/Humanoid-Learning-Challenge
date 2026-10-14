"""Scripted oracle over the 50 stored init states and 50 wider random placements (tests set_obj_pose too)."""

import json
import sys

import numpy as np
from oracle_pick_place import plan

from handsim.sim import follow, make_env, obj_pose, set_obj_pose


def main(out_dir: str, n_wide: int = 50) -> None:
    rows = []
    for i in range(50):
        env, _, obs = make_env(init=i)
        _, ok, _ = follow(env, obs, plan(obs, 0.04, -0.05))
        rows.append({"kind": "stored", "i": i, "success": ok})
        env.close()
    rng = np.random.default_rng(0)
    for i in range(n_wide):
        env, _, obs = make_env(init=0)
        bowl, bq = obj_pose(obs, "akita_black_bowl_1")
        plate, pq = obj_pose(obs, "plate_1")
        while True:
            b = bowl + [*rng.uniform(-0.10, 0.10, 2), 0.01]
            p = plate + [*rng.uniform(-0.10, 0.10, 2), 0.01]
            if np.linalg.norm((b - p)[:2]) > 0.16:
                break
        set_obj_pose(env, "plate_1", p, pq, settle=0)
        obs = set_obj_pose(env, "akita_black_bowl_1", b, bq)
        _, ok, _ = follow(env, obs, plan(obs, 0.04, -0.05))
        rows.append({"kind": "wide", "i": i, "bowl": b.tolist(), "plate": p.tolist(), "success": ok})
        env.close()
    summary = {k: float(np.mean([r["success"] for r in rows if r["kind"] == k])) for k in ("stored", "wide")}
    with open(f"{out_dir}/robustness.json", "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)
    print(summary)


if __name__ == "__main__":
    main(sys.argv[1])
