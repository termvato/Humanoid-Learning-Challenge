import numpy as np

from handsim.augment import centred_place_shift, release_early, retime, shift_profile, warp
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


def test_retime_drops_pauses_keeps_grip_dwell():
    eef = np.zeros((40, 3))
    eef[10:20, 0] = np.linspace(0.01, 0.1, 10)  # move, then a long pause from 20 on
    eef[20:, 0] = 0.1
    grip = np.full(40, GRIP_OPEN)
    grip[30:] = GRIP_CLOSE
    idx = retime(eef, grip, min_step=0.002, dwell=5)
    assert list(idx[:2]) == [0, 10]  # the pause before the move is gone
    assert set(range(10, 20)) <= set(idx) and set(range(30, 35)) <= set(idx) and 25 not in idx
    assert idx[-1] == 39


def test_centred_place_shift_without_grasp_is_zero():
    assert np.allclose(centred_place_shift(np.zeros((5, 3)), np.full(5, GRIP_OPEN), np.zeros(2), np.ones(2)), 0)


def test_release_early_opens_above_the_set_down_height():
    eef = np.zeros((40, 3))
    eef[:, 2] = 0.2
    eef[25:31, 2] = np.linspace(0.2, 0.1, 6)  # final descent, released at step 31
    eef[31:, 2] = 0.1
    grip = np.full(40, GRIP_OPEN)
    grip[10:31] = GRIP_CLOSE
    out = release_early(eef, grip, 0.02)
    k = np.flatnonzero(out[10:] != GRIP_CLOSE)[0] + 10
    assert eef[k, 2] <= 0.12 + 1e-9 < eef[k - 1, 2] and k < 31 and np.all(out[10:k] == GRIP_CLOSE)


def test_release_early_keeps_the_grasp_on_a_low_carry():
    eef = np.zeros((30, 3))
    eef[10:20, 2] = 0.01  # carried only 1 cm above the set-down height
    grip = np.full(30, GRIP_OPEN)
    grip[5:20] = GRIP_CLOSE
    out = release_early(eef, grip, 0.02)
    assert out[5] == GRIP_CLOSE and out[12] == GRIP_CLOSE
