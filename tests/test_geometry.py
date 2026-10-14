import numpy as np
from scipy.spatial.transform import Rotation as R

from handsim.geometry import (
    fill_gaps,
    fit_plane,
    grip_closed,
    gripper_pose,
    objects_on_table,
    real_to_sim,
    solve_translation,
    table_frame,
)


def test_fit_plane_recovers_tilted_table_with_clutter():
    rng = np.random.default_rng(1)
    n_true = R.from_euler("x", 5, degrees=True).apply([0, 1, 0])  # ARKit world: +y up
    xy = rng.uniform(-0.5, 0.5, (2000, 2))
    basis = np.linalg.svd(n_true[None])[2][1:]
    pts = xy @ basis + 0.7 * n_true + rng.normal(0, 0.001, (2000, 1)) * n_true
    clutter = rng.uniform(-0.5, 0.5, (300, 3))
    n, d = fit_plane(np.vstack([pts, clutter]), up=np.array([0, 1, 0]))
    assert np.degrees(np.arccos(n @ n_true)) < 0.5
    assert abs(d + 0.7) < 0.003


def test_table_frame_puts_origin_on_plane():
    T = table_frame(np.array([0, 1.0, 0]), -0.7, np.array([0.3, 0.9, 0.1]), np.array([1.0, 0.2, 0]))
    assert np.allclose(T[:3, 3], [0.3, 0.7, 0.1])
    assert np.allclose(T[:3, 2], [0, 1, 0])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3))


def test_objects_on_table_finds_bowl_and_plate():
    rng = np.random.default_rng(0)
    table = np.c_[rng.uniform(-0.3, 0.6, (20000, 2)), rng.normal(0, 0.001, 20000)]
    a = rng.uniform(0, 2 * np.pi, 4000)
    r = 0.055 * np.sqrt(rng.uniform(0, 1, 4000))
    bowl = np.c_[r * np.cos(a), r * np.sin(a), rng.uniform(0.01, 0.053, 4000)]
    plate = np.c_[0.28 + 1.27 * r * np.cos(a), 1.27 * r * np.sin(a), np.full(4000, 0.015)]
    objs = objects_on_table(np.vstack([table, bowl, plate]))
    assert len(objs) == 2
    tall = max(objs, key=lambda o: o["height"])
    assert np.linalg.norm(tall["centre"]) < 0.01 and abs(tall["height"] - 0.05) < 0.01


def test_solve_translation_inverts_projection():
    K = np.array([[1400.0, 0, 960], [0, 1400, 720], [0, 0, 1]])
    kp = np.random.default_rng(0).normal(0, 0.04, (21, 3))
    t = np.array([0.05, -0.1, 0.45])
    p = kp + t
    uv = (p[:, :2] / p[:, 2:]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    assert np.allclose(solve_translation(kp, uv, K), t, atol=1e-6)


def test_gripper_pose_axes():
    kp = np.zeros((21, 3))
    kp[0] = [0, 0, 0.1]  # wrist above
    kp[2] = [0, -0.02, 0.0]  # thumb MCP
    kp[5] = [0, 0.02, 0.0]  # index MCP
    kp[4] = [0, -0.01, -0.05]
    kp[8] = [0, 0.01, -0.05]
    mid, rot, w = gripper_pose(kp)
    assert np.allclose(mid, [0, 0, -0.05]) and abs(w - 0.02) < 1e-9
    assert np.allclose(rot[:, 1], [0, 1, 0]) and np.allclose(rot[:, 2], [0, 0, -1])
    assert np.isclose(np.linalg.det(rot), 1)


def test_fill_gaps_short_only():
    x = np.arange(20.0)[:, None]
    v = np.ones(20, bool)
    v[3:5] = False
    v[8:20] = False
    v[19] = True
    f, ok = fill_gaps(x * 0 + x, v, max_gap=3)
    assert ok[3] and ok[4] and np.allclose(f[3:5, 0], [3, 4])
    assert not ok[10]


def test_grip_closed_hysteresis():
    w = np.r_[np.full(20, 0.08), np.full(20, 0.01), np.full(20, 0.08)]
    w[25] = 0.05  # one noisy frame mid-grasp must not open
    closed, _ = grip_closed(w)
    assert not closed[:20].any() and closed[20:40].all() and not closed[40:].any()


def test_real_to_sim_maps_objects():
    T = real_to_sim(np.array([0.1, 0.2]), np.array([0.1, 0.5]), np.array([0, 0, 0.9]), np.array([0.3, 0, 0.9]))
    bowl = T @ [0.1, 0.2, 0, 1]
    plate = T @ [0.1, 0.5, 0, 1]
    assert np.allclose(bowl[:3], [0, 0, 0.9]) and np.allclose(plate[:3], [0.3, 0, 0.9])
