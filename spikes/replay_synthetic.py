"""Synthetic hand that does the oracle's rim grasp in a made-up real frame (plate along +x): tests replay.py's
mapping, smoothing, grip detection and following without a recording."""

import json
import sys
from pathlib import Path

import numpy as np


def keypoints(mid, rot, w):
    _x, y, z = rot.T
    kp = np.zeros((21, 3))
    kp[4], kp[8] = mid - y * w / 2, mid + y * w / 2
    kp[2], kp[5] = mid - y * 0.02 - z * 0.06, mid + y * 0.02 - z * 0.06
    kp[0] = mid - z * 0.12
    return kp


def main(out: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bowl, plate = np.array([0.1, 0.05]), np.array([0.36, 0.05])  # 26 cm apart along +x
    rot = np.stack([[0, -1, 0], [1, 0, 0], [0, 0, -1]], 1).astype(float)  # closing axis +x, pointing down
    grasp = np.r_[bowl - [0.05, 0], 0.029]
    place = np.r_[plate - [0.05, 0], 0.049]
    keys = [  # (time s, tip midpoint, width)
        (0.0, grasp + [0, 0, 0.12], 0.08), (1.5, grasp, 0.08), (2.0, grasp, 0.01), (3.0, grasp + [0, 0, 0.12], 0.01),
        (4.5, place + [0, 0, 0.12], 0.01), (5.5, place, 0.01), (6.0, place, 0.08), (7.0, place + [0, 0, 0.12], 0.08),
    ]
    t = np.arange(0, 7.0, 1 / 30)
    kt = [k[0] for k in keys]
    mids = np.stack([np.interp(t, kt, [k[1][d] for k in keys]) for d in range(3)], 1)
    widths = np.interp(t, kt, [k[2] for k in keys])
    kp = np.stack([keypoints(m, rot, w) for m, w in zip(mids, widths, strict=True)])
    kp += np.random.default_rng(0).normal(0, 0.003, kp.shape)  # 3 mm tracking noise
    n = len(t)
    np.savez(out / "hand.npz", kp_world=kp, kp_2d=np.zeros((n, 21, 2)), found=np.ones(n, bool),
             depth_ok=np.ones(n, bool), depth_offset=np.zeros(n), rows=np.arange(n), t=t)
    scene = {"T_world_table": np.eye(4).tolist(),
             "bowl": {"centre": bowl.tolist(), "height": 0.053, "diameter": 0.11},
             "plate": {"centre": plate.tolist(), "height": 0.019, "diameter": 0.14}}
    (out / "scene.json").write_text(json.dumps(scene))


if __name__ == "__main__":
    main(sys.argv[1])
