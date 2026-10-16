"""Remove the human's pauses from a retargeted source track (see handsim.augment.retime).

Usage: .venv/bin/python tools/retime.py outputs/take04/traj_sim.npz outputs/take04/traj_sim_retimed.npz [--hold-wrist] [--release-early]

--release-early opens the gripper 2 cm above the human's set-down height (see handsim.augment.release_early).
--hold-wrist keeps the gripper at its orientation at the grasp for the whole track: the human's wrist turns are
the main difference from LIBERO's own demos, where the wrist barely turns.
"""

import sys

import numpy as np

from handsim.augment import release_early, retime


def main(src: str, dst: str, hold_wrist: bool = False, early: bool = False) -> None:
    z = dict(np.load(src))
    if early:
        z["grip"] = release_early(z["eef"], z["grip"])
    idx = retime(z["eef"], z["grip"])
    for k in ("eef", "quat", "grip"):
        z[k] = z[k][idx]
    if hold_wrist:
        grasp = np.flatnonzero(np.diff(np.sign(z["grip"])) > 0)
        z["quat"] = np.tile(z["quat"][grasp[-1] + 1 if len(grasp) else 0], (len(idx), 1))
    z["t"] = np.arange(len(idx)) / 20
    np.savez(dst, **z)
    print(src, len(np.load(src)["eef"]), "->", len(idx), "steps")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--hold-wrist" in sys.argv, "--release-early" in sys.argv)
