"""README figures from the real outputs (run with .venv). Writes docs/media/.

Usage: .venv/bin/python tools/make_figures.py
"""

import csv
import json
from pathlib import Path

import cv2
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/media")


def takes_sheet() -> None:
    """Phone frames: start, grasp, carry, place for the pinch take."""
    v = cv2.VideoCapture("data/raw/take04/rgb.mp4")
    frames = []
    while True:
        ok, im = v.read()
        if not ok:
            break
        frames.append(cv2.resize(im, (320, 240)))
    idx = np.linspace(0, len(frames) - 1, 6).astype(int)
    cv2.imwrite(
        str(OUT / "take04_frames.jpg"), np.hstack([frames[i] for i in idx]), [cv2.IMWRITE_JPEG_QUALITY, 80]
    )


def side_by_side_gif() -> None:
    fr = iio.imread("outputs/take04/side_by_side.mp4")[::3]
    fr = [cv2.resize(f, (f.shape[1] * 160 // f.shape[0], 160)) for f in fr]
    iio.imwrite(OUT / "take04_phone_vs_sim.gif", fr, duration=150, loop=0)


def scene_top() -> None:
    im = cv2.imread("outputs/take04/scene_top.png")
    cv2.imwrite(str(OUT / "take04_scene_top.png"), im)


def rim_sweep() -> None:
    rows = json.loads(Path("outputs/spikes/rim_angle_sweep.json").read_text())
    rows += json.loads(Path("outputs/spikes/rim_angle_sweep_60_150.json").read_text())
    by = {r["angle_deg"]: r["success"] for r in rows}
    a = np.array(sorted(by))
    ok = np.array([by[x] for x in a])
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection="polar")
    ax.scatter(np.radians(a[ok]), np.ones(ok.sum()), c="tab:green", label="lifted")
    ax.scatter(np.radians(a[~ok]), np.ones((~ok).sum()), c="tab:red", marker="x", label="failed (handle)")
    ax.set_yticks([])
    ax.set_title("Scripted rim pinch around the sim mug")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=8)
    fig.savefig(OUT / "rim_sweep.png", dpi=110, bbox_inches="tight")


def results() -> None:
    rows = list(csv.DictReader(Path("outputs/results.csv").read_text().splitlines()))
    act = [r for r in rows if r["name"].startswith("act_clean40k") or r["name"].startswith("act_real3_")]
    pts = {}
    for r in act:
        k = int(r["name"].rstrip("k").split("_")[-1])
        pts.setdefault(k, []).append((float(r["success_pct"]), int(r["episodes"])))
    ks = sorted(pts)
    y = [np.average([p for p, _ in pts[k]], weights=[n for _, n in pts[k]]) for k in ks]
    n = [sum(n for _, n in pts[k]) for k in ks]
    err = [196 * np.sqrt(max(v / 100 * (1 - v / 100), 1e-4) / m) for v, m in zip(y, n, strict=True)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.errorbar(ks, y, yerr=err, marker="o", capsize=3)
    ax.set_xlabel("ACT training steps (thousands)")
    ax.set_ylabel("success on libero_90 #72 (%)")
    ax.set_title("ACT trained on demos from 3 phone takes (95% CI)")
    ax.set_ylim(0, max(50, max(y) + 15))
    fig.savefig(OUT / "act_training_curve.png", dpi=110, bbox_inches="tight")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (takes_sheet, side_by_side_gif, scene_top, rim_sweep, results):
        f()
        print("ok", f.__name__)
