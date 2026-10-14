"""Stray capture -> table frame, pick and place object positions (scene.json), plus the S2 checks and a top view.

Pick object = the tallest blob (mug, bowl); place object = the widest of the rest (plate, or any flat stand-in).

Uses the first frames (hand not yet in view) for objects, the last frames to check that the table stays put.

Usage: .venv/bin/python tools/scene.py data/raw/take01 outputs/take01 [--first 30] [--last 30]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from handsim.geometry import fit_plane, objects_on_table, table_frame
from handsim.io.stray import backproject, load_stray

UP = np.array([0.0, 1.0, 0.0])  # ARKit world: +y against gravity


def cloud(cap, idx, max_depth: float = 1.2) -> np.ndarray:
    return np.vstack([
        backproject(cap.depth(i), cap.K_depth(i), cap.T_wc[i], cap.confidence(i), max_depth=max_depth) for i in idx
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--first", type=int, default=30, help="frames at the start with no hand in view")
    ap.add_argument("--last", type=int, default=30)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = load_stray(args.capture)
    n = len(cap)

    start = cloud(cap, range(0, min(args.first, n), 3))
    normal, d = fit_plane(start, UP)
    on_plane = np.abs(start @ normal + d) < 0.01
    x_hint = np.array([1.0, 0, 0])
    T_wt = table_frame(normal, d, start[on_plane].mean(0), x_hint)
    T_tw = np.linalg.inv(T_wt)
    pts_t = start @ T_tw[:3, :3].T + T_tw[:3, 3]
    objs = objects_on_table(pts_t)
    cand = [o for o in objs if 0.008 < o["height"] < 0.2 and 0.04 < o["diameter"] < 0.40]
    pick = max(cand, key=lambda o: o["height"]) if cand else None
    rest = [o for o in cand if o is not pick]
    place = max(rest, key=lambda o: o["diameter"]) if rest else None

    end = cloud(cap, range(max(0, n - args.last), n, 3))
    end_on = end[np.abs(end @ normal + d) < 0.02]
    drift = float(np.median(np.abs(end_on @ normal + d))) if len(end_on) else float("nan")

    res = {
        "T_world_table": T_wt.tolist(),
        "objects": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items()} for o in objs],
        "pick": None if pick is None else {**pick, "centre": pick["centre"].tolist()},
        "place": None if place is None else {**place, "centre": place["centre"].tolist()},
        "s2_table_drift_m": drift,
        "s2_pick_place_m": None if not (pick and place) else float(np.linalg.norm(pick["centre"] - place["centre"])),
    }
    with open(out / "scene.json", "w") as f:
        json.dump(res, f, indent=1)

    sel = (pts_t[:, 2] > -0.02) & (pts_t[:, 2] < 0.2)
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pts_t[sel, 0], pts_t[sel, 1], c=pts_t[sel, 2] * 100, s=0.5, cmap="viridis", vmin=0, vmax=8)
    for name, o in (("pick", pick), ("place", place)):
        if o:
            ax.add_patch(plt.Circle(o["centre"], o["diameter"] / 2, fill=False, color="r"))
            ax.text(*o["centre"], f"{name}\n{o['height'] * 100:.1f} cm high", color="r", ha="center")
    ax.set_aspect("equal")
    ax.set_xlabel("table x (m)")
    ax.set_ylabel("table y (m)")
    fig.colorbar(sc, label="height above table (cm)")
    ax.set_title(f"table drift {drift * 100:.2f} cm, pick-place {res['s2_pick_place_m'] or float('nan'):.3f} m")
    fig.savefig(out / "scene_top.png", dpi=120, bbox_inches="tight")
    print(json.dumps({k: res[k] for k in ("pick", "place", "s2_table_drift_m", "s2_pick_place_m")}, indent=1))


if __name__ == "__main__":
    main()
