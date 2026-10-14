"""Retargeted demo(s) -> N successful sim demos at random object placements -> LeRobotDataset.

Usage: .venv/bin/python tools/generate_demos.py outputs/take01/traj_sim.npz [more traj files] \
           --n 200 --radius 0.10 --out data/demos/take01_r10 [--workers 24]
Writes the dataset plus attempts.json (every placement tried and whether it succeeded).
"""

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np


def sample(rng, pick0, place0, radius: float, min_gap: float = 0.16):
    while True:
        a = pick0 + rng.uniform(-radius, radius, 2)
        b = place0 + rng.uniform(-radius, radius, 2)
        if np.linalg.norm(a - b) > min_gap:
            return a, b


def worker(args) -> list:
    task_name, trajs, seeds, radius, tmp = args
    from handsim.augment import place, rollout, warp
    from handsim.sim import TASKS, make_env

    task = TASKS[task_name]
    env, _, _ = make_env(task_id=task.task_id)
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        tr = trajs[s % len(trajs)]
        a, b = sample(rng, tr["pick"][:2], tr["place"][:2], radius)
        env.reset()
        obs = place(env, task, a, b)
        eef = warp(tr["eef"], tr["grip"], np.r_[a - tr["pick"][:2], 0], np.r_[b - tr["place"][:2], 0])
        ok, frames = rollout(env, obs, eef, tr["quat"], tr["grip"], record=True)
        row = {"seed": int(s), "source": int(s % len(trajs)), "pick": a.tolist(), "place": b.tolist(),
               "success": bool(ok), "steps": len(frames)}
        if ok:
            path = Path(tmp) / f"ep_{s:06d}.npz"
            np.savez(path, **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
            row["file"] = str(path)
        out.append(row)
    env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", nargs="+")
    ap.add_argument("--n", type=int, default=200, help="successful demos wanted")
    ap.add_argument("--radius", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--task", default="mug", help="key of handsim.sim.TASKS")
    args = ap.parse_args()
    out = Path(args.out)
    tmp = out.parent / (out.name + "_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    trajs = [dict(np.load(t)) for t in args.traj]

    rows, next_seed = [], 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        while sum(r["success"] for r in rows) < args.n:
            need = args.n - sum(r["success"] for r in rows)
            rate = max(0.2, np.mean([r["success"] for r in rows])) if rows else 0.5
            batch = int(need / rate * 1.1) + args.workers
            seeds = np.arange(next_seed, next_seed + batch)
            next_seed += batch
            chunks = [(args.task, trajs, seeds[w :: args.workers].tolist(), args.radius, str(tmp))
                      for w in range(args.workers)]
            for res in pool.imap_unordered(worker, chunks):
                rows += res
            print(f"tried {len(rows)}, success {sum(r['success'] for r in rows)}", flush=True)
            if len(rows) > 20 * args.n:
                break

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from handsim.sim import TASKS

    language = TASKS[args.task].language

    feats = {
        "observation.images.image": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.images.image2": {"dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=f"local/{out.name}", fps=20, features=feats, root=out, robot_type="panda",
                               use_videos=True, image_writer_threads=8)
    kept = [r for r in rows if r["success"]][: args.n]
    for r in kept:
        ep = np.load(r["file"])
        for i in range(len(ep["action"])):
            ds.add_frame({k: ep[k][i] for k in feats} | {"task": language})
        ds.save_episode()
    ds.finalize()
    shutil.rmtree(tmp)
    summary = {"tried": len(rows), "success": sum(r["success"] for r in rows),
               "success_rate": float(np.mean([r["success"] for r in rows])), "kept": len(kept),
               "radius": args.radius, "sources": args.traj, "task": args.task}
    (out / "attempts.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
