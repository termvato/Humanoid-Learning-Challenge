"""S1: step and render speed of one LIBERO task on this laptop."""

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main(out_dir: str) -> None:
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    res = {}
    for size in (256, 128):
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
        env.seed(0)
        env.reset()
        env.set_init_state(suite.get_task_init_states(0)[0])
        t = time.time()
        for _ in range(200):
            obs, *_ = env.step(np.random.uniform(-1, 1, 7))
        res[f"fps_{size}"] = 200 / (time.time() - t)
        res["obs_keys"] = sorted(obs.keys())
        env.close()
    res["task"] = task.language
    os.makedirs(out_dir, exist_ok=True)
    import imageio

    imageio.imwrite(os.path.join(out_dir, "agentview.png"), obs["agentview_image"][::-1])
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
