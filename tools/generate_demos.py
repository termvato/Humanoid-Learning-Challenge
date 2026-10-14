"""Retargeted demo(s) -> N successful sim demos at random object placements -> LeRobotDataset.

Usage: .venv/bin/python tools/generate_demos.py outputs/take01/traj_sim.npz [more traj files] \
           --n 200 --radius 0.10 --out data/demos/take01_r10 [--workers 24]
Writes the dataset plus attempts.json (every placement tried and whether it succeeded).
"""

import argparse
import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

CHUNK = 8  # rollouts per task; each task gets a fresh process (env.close() leaks ~1.5 GB)


def sample(rng, pick0, place0, radius: float, min_gap: float = 0.16):
    while True:
        a = pick0 + rng.uniform(-radius, radius, 2)
        b = place0 + rng.uniform(-radius, radius, 2)
        if np.linalg.norm(a - b) > min_gap:
            return a, b


def worker(args) -> list:
    task_name, trajs, seeds, radius, tmp, around, centre, noise = args
    from handsim.augment import centred_place_shift, place, rollout, warp
    from handsim.sim import TASKS, make_env, obj_pose

    task = TASKS[task_name]
    env, _, obs0 = make_env(task_id=task.task_id)
    default = obj_pose(obs0, task.pick)[0][:2], obj_pose(obs0, task.place)[0][:2]
    out = []
    for s in seeds:
        rng = np.random.default_rng(s)
        tr = trajs[s % len(trajs)]
        centres = default if around == "default" else (tr["pick"][:2], tr["place"][:2])
        a, b = sample(rng, *centres, radius)
        env.reset()
        obs = place(env, task, a, b)
        d_pick = np.r_[a - tr["pick"][:2], 0]
        if centre:
            # the grasp moves with the object (d_pick), so the held object starts at a; centre it on b
            d_place = centred_place_shift(tr["eef"] + d_pick, tr["grip"], a, b) + d_pick
        else:
            d_place = np.r_[b - tr["place"][:2], 0]
        eef = warp(tr["eef"], tr["grip"], d_pick, d_place)
        ok, frames = rollout(env, obs, eef, tr["quat"], tr["grip"], record=True, noise=noise, rng=rng)
        row = {
            "seed": int(s),
            "source": int(s % len(trajs)),
            "pick": a.tolist(),
            "place": b.tolist(),
            "success": bool(ok),
            "steps": len(frames),
        }
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
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="each renders on the GPU (~2 GB RAM + a GL context); 14+ ran out of memory",
    )
    ap.add_argument("--task", default="mug", help="key of handsim.sim.TASKS")
    ap.add_argument(
        "--around",
        choices=["default", "traj"],
        default="default",
        help="sample placements around the task's default object positions (what LIBERO evaluates) or the replay's",
    )
    ap.add_argument(
        "--no-centre", action="store_true", help="keep the human's release point instead of centring"
    )
    ap.add_argument("--noise", type=float, default=0.0, help="DART action noise std (normalised units)")
    args = ap.parse_args()
    out = Path(args.out)
    tmp = out.parent / (out.name + "_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    trajs = [dict(np.load(t)) for t in args.traj]

    rows, next_seed = [], 0
    # small chunks in a process pool: a worker that dies (seen with MuJoCo/EGL) loses only its chunk, which is
    # logged as failed, and the pool is rebuilt
    ctx = mp.get_context("spawn")
    pool = ProcessPoolExecutor(args.workers, mp_context=ctx, max_tasks_per_child=1)
    while sum(r["success"] for r in rows) < args.n and len(rows) <= 20 * args.n:
        need = args.n - sum(r["success"] for r in rows)
        rate = max(0.2, np.mean([r["success"] for r in rows])) if rows else 0.5
        batch = int(need / rate * 1.1) + args.workers
        seeds = np.arange(next_seed, next_seed + batch).tolist()
        next_seed += batch
        chunks = [seeds[k : k + CHUNK] for k in range(0, len(seeds), CHUNK)]
        futs = {
            pool.submit(
                worker,
                (args.task, trajs, c, args.radius, str(tmp), args.around, not args.no_centre, args.noise),
            ): c
            for c in chunks
        }
        broken = False
        for f in as_completed(futs):
            try:
                rows += f.result()
            except Exception as e:  # noqa: BLE001
                print(f"chunk {futs[f][0]}: {type(e).__name__}: {e}", flush=True)
                broken = broken or isinstance(e, BrokenProcessPool)
                rows += [{"seed": int(x), "success": False, "error": type(e).__name__} for x in futs[f]]
        if broken:
            pool.shutdown(cancel_futures=True)
            pool = ProcessPoolExecutor(args.workers, mp_context=ctx, max_tasks_per_child=1)
        print(f"tried {len(rows)}, success {sum(r['success'] for r in rows)}", flush=True)
    pool.shutdown()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from handsim.sim import TASKS

    language = TASKS[args.task].language

    feats = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.image2": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(
        repo_id=f"local/{out.name}",
        fps=20,
        features=feats,
        root=out,
        robot_type="panda",
        use_videos=True,
        image_writer_threads=8,
    )
    kept = [r for r in rows if r["success"]][: args.n]
    for r in kept:
        ep = np.load(r["file"])
        for i in range(len(ep["action"])):
            ds.add_frame({k: ep[k][i] for k in feats} | {"task": language})
        ds.save_episode()
    ds.finalize()
    shutil.rmtree(tmp)
    summary = {
        "tried": len(rows),
        "success": sum(r["success"] for r in rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "worker_errors": sum("error" in r for r in rows),
        "kept": len(kept),
        "radius": args.radius,
        "sources": args.traj,
        "task": args.task,
        "around": args.around,
        "centre_on_place": not args.no_centre,
        "noise": args.noise,
    }
    (out / "attempts.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
