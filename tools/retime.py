"""Remove the human's pauses from a retargeted source track (see handsim.augment.retime).

Usage: .venv/bin/python tools/retime.py outputs/take04/traj_sim.npz outputs/take04/traj_sim_retimed.npz
"""

import sys

import numpy as np

from handsim.augment import retime


def main(src: str, dst: str) -> None:
    z = dict(np.load(src))
    idx = retime(z["eef"], z["grip"])
    for k in ("eef", "quat", "grip"):
        z[k] = z[k][idx]
    z["t"] = np.arange(len(idx)) / 20
    np.savez(dst, **z)
    print(src, len(np.load(src)["eef"]), "->", len(idx), "steps")


if __name__ == "__main__":
    main(*sys.argv[1:3])
