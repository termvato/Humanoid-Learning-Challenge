import numpy as np

from handsim.augment import centred_place_shift, shift_profile, warp
from handsim.sim import GRIP_CLOSE, GRIP_OPEN


def test_warp_follows_pick_then_place():
    grip = np.array([GRIP_OPEN] * 3 + [GRIP_CLOSE] * 4 + [GRIP_OPEN] * 3)
    w = shift_profile(grip)
    assert w[:4].max() == 0 and np.all(w[7:] == 1)
    out = warp(np.zeros((10, 3)), grip, np.array([1.0, 0, 0]), np.array([0, 2.0, 0]))
    assert np.allclose(out[0], [1, 0, 0]) and np.allclose(out[-1], [0, 2, 0])


def test_centred_place_shift_puts_held_object_on_target():
    eef = np.zeros((10, 3))
    eef[:, 0] = np.linspace(0, 0.3, 10)
    grip = np.full(10, GRIP_OPEN)
    grip[3:8] = GRIP_CLOSE
    pick = eef[3, :2] + [0.04, 0.0]  # object centre 4 cm from the gripper at the grasp
    target = np.array([0.25, 0.05])
    s = centred_place_shift(eef, grip, pick, target)
    assert np.allclose(eef[7, :2] + s[:2] + [0.04, 0.0], target) and s[2] == 0
