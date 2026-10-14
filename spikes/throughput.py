"""Plan 2 gate: env steps/s with many processes and cameras off (rule: >= 2000 -> PPO is an option, else SAC)."""

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np


def worker(n_steps: int, q) -> None:
    os.environ["MUJOCO_GL"] = "egl"
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    b = benchmark.get_benchmark_dict()["libero_90"]()
    t = b.get_task(9)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, has_offscreen_renderer=False, use_camera_obs=False)
    env.reset()
    t0 = time.time()
    for _ in range(n_steps):
        env.step(np.random.uniform(-1, 1, 7))
    q.put(n_steps / (time.time() - t0))


def main(out: str) -> None:
    res = {}
    for n in (1, 8, 16, 24, 30):
        q = mp.Queue()
        ps = [mp.Process(target=worker, args=(300, q)) for _ in range(n)]
        for p in ps:
            p.start()
        rates = [q.get() for _ in ps]
        for p in ps:
            p.join()
        res[n] = sum(rates)
        print(n, "procs:", round(res[n]), "steps/s", flush=True)
    with open(out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main(sys.argv[1])
