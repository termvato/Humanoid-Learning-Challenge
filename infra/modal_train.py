"""Policy training on Modal: lerobot-train on a cloud GPU, so the laptop stays free for sim work.

The volume `handsim` holds datasets (/vol/demos/<name>, uploaded from data/demos) and runs (/vol/train/<run>).
The image has LeRobot only, no simulator: training needs the dataset, evaluation stays on the laptop.

Usage:
  .venv/bin/modal volume create handsim                                  # once
  .venv/bin/modal volume put handsim data/demos/real3_r04_n02 /demos/real3_r04_n02
  .venv/bin/modal run --detach infra/modal_train.py --dataset real3_r04_n02 --run act_n02 \
      --args "--policy.type=act --policy.chunk_size=50 --policy.n_action_steps=20 --batch_size=32 --steps=20000"
  .venv/bin/modal volume get handsim /train/act_n02/checkpoints/last outputs/train/act_n02/
"""

import shlex
import subprocess
import time
from pathlib import Path

import modal

VOL = Path("/vol")

vol = modal.Volume.from_name("handsim", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install("lerobot[dataset,training,smolvla]==0.6.1")
)
app = modal.App("handsim-train")


@app.function(image=image, gpu="L4", cpu=8.0, memory=32768, timeout=6 * 3600, volumes={"/vol": vol})
def train(dataset: str, run: str, args: str) -> str:
    out = VOL / "train" / run
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id=local/{dataset}",
        f"--dataset.root={VOL / 'demos' / dataset}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        "--env_eval_freq=0",
        "--wandb.enable=false",
        "--num_workers=8",
        "--log_freq=500",
        "--save_freq=5000",
        f"--output_dir={out}",
        *shlex.split(args),
    ]
    print(" ".join(cmd), flush=True)
    log = VOL / "train" / f"{run}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:  # commit checkpoints as they appear, so a killed job keeps them
            time.sleep(600)
            vol.commit()
    vol.commit()
    tail = log.read_text()[-3000:]
    if p.returncode:
        raise RuntimeError(f"lerobot-train failed ({p.returncode}):\n{tail}")
    return tail


@app.local_entrypoint()
def main(dataset: str, run: str, args: str) -> None:
    print(train.remote(dataset, run, args))
