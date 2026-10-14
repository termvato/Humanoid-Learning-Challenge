"""Geometry shared by the hand and sim sides: table plane, objects on it, gripper frame from hand keypoints,
and the real-to-sim mapping. Numpy and scipy only, so it imports in both envs.

Hand keypoints use the MANO / OpenPose order: 0 wrist, 1-4 thumb (CMC, MCP, IP, tip), 5-8 index (MCP ... tip).
Gripper frame follows the Panda end effector: z = approach, y = finger closing axis, x = y cross z.
"""

import itertools

import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

WRIST, THUMB_MCP, THUMB_TIP, INDEX_MCP, INDEX_TIP = 0, 2, 4, 5, 8
PANDA_TIP_OFFSET = 0.011  # m, finger pads sit this far beyond the eef site along z (measured in LIBERO)


def fit_plane(points: np.ndarray, up: np.ndarray, iters: int = 300, tol: float = 0.005, seed: int = 0):
    """RANSAC plane through (N,3) points. Returns (normal, d) with normal . p + d = 0, normal along `up`."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ up) < 0.9:  # a table is roughly level
            continue
        inl = np.abs(points @ n - n @ a) < tol
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    p = points[best]
    centre = p.mean(0)
    n = np.linalg.svd(p - centre)[2][-1]  # least-squares refit on inliers
    if n @ up < 0:
        n = -n
    return n, -n @ centre


def table_frame(normal: np.ndarray, d: float, origin_hint: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    """4x4 world-from-table: z = table normal, origin = hint projected onto the plane, x = hint direction in plane."""
    z = normal / np.linalg.norm(normal)
    o = origin_hint - (z @ origin_hint + d) * z
    x = x_hint - (x_hint @ z) * z
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(z, x), z], 1)
    T[:3, 3] = o
    return T


def objects_on_table(pts_t: np.ndarray, cell: float = 0.005, min_h: float = 0.008, min_cells: int = 40):
    """Blobs above the table from table-frame points. Returns dicts (centre xy, height, diameter), largest first."""
    p = pts_t[(pts_t[:, 2] > min_h) & (pts_t[:, 2] < 0.4)]
    if len(p) == 0:
        return []
    ij = np.floor(p[:, :2] / cell).astype(int)
    lo = ij.min(0)
    grid = np.zeros(ij.max(0) - lo + 1, bool)
    grid[tuple((ij - lo).T)] = True
    grid = ndimage.binary_closing(grid, iterations=2)
    lab, n = ndimage.label(grid)
    cell_lab = lab[tuple((ij - lo).T)]
    out = []
    for k in range(1, n + 1):
        cells = np.argwhere(lab == k)
        if len(cells) < min_cells:
            continue
        q = p[cell_lab == k]
        xy = (cells + lo + 0.5) * cell
        out.append({
            "centre": xy.mean(0),
            "height": float(np.percentile(q[:, 2], 95)) if len(q) else 0.0,
            "diameter": float(np.ptp(xy, 0).max() + cell),
            "cells": len(cells),
        })
    return sorted(out, key=lambda o: -o["cells"])


def solve_translation(kp_rel: np.ndarray, uv: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame translation t so that kp_rel + t projects onto uv with intrinsics K (linear least squares)."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    du, dv = uv[:, 0] - cx, uv[:, 1] - cy
    X, Y, Z = kp_rel.T
    zeros = np.zeros_like(du)
    A = np.concatenate([np.stack([fx + zeros, zeros, -du], 1), np.stack([zeros, fy + zeros, -dv], 1)])
    b = np.concatenate([du * Z - fx * X, dv * Z - fy * Y])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def gripper_pose(kp: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(tip midpoint, 3x3 rotation, thumb-index tip distance) from (21,3) keypoints in one frame.

    Axes from the knuckles (stable when the fingers pinch), as HumanEgo does.
    """
    y = kp[INDEX_MCP] - kp[THUMB_MCP]
    y /= np.linalg.norm(y)
    z = (kp[INDEX_MCP] + kp[THUMB_MCP]) / 2 - kp[WRIST]
    z -= (z @ y) * y
    z /= np.linalg.norm(z)
    mid = (kp[THUMB_TIP] + kp[INDEX_TIP]) / 2
    return mid, np.stack([np.cross(y, z), y, z], 1), float(np.linalg.norm(kp[THUMB_TIP] - kp[INDEX_TIP]))


def fill_gaps(x: np.ndarray, valid: np.ndarray, max_gap: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Linear fill of invalid runs up to max_gap long (per column). Returns (filled, still_valid)."""
    x = x.copy()
    idx = np.flatnonzero(valid)
    ok = valid.copy()
    for a, b in itertools.pairwise(idx):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            x[a + 1 : b] = x[a] * (1 - w[:, None]) + x[b] * w[:, None]
            ok[a + 1 : b] = True
    return x, ok


def smooth_track(pos: np.ndarray, rot: np.ndarray, width: np.ndarray, fps: float):
    """HumanEgo-style smoothing: Savitzky-Golay on position (21 frames at 30 Hz, scaled to fps),
    EMA on the rotation axes then re-orthonormalised, median filter on width."""
    win = max(5, round(21 * fps / 30) | 1)
    pos = savgol_filter(pos, min(win, len(pos) - (1 - len(pos) % 2)), 2, axis=0) if len(pos) > 5 else pos
    alpha = 1 - (1 - 0.15) ** (30 / fps)
    out = rot.copy()
    for t in range(1, len(rot)):
        m = (1 - alpha) * out[t - 1] + alpha * rot[t]
        u, _, vt = np.linalg.svd(m)
        out[t] = u @ vt
    width = ndimage.median_filter(width, size=max(3, int(fps / 6) | 1))
    return pos, out, width


def grip_closed(width: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed where the pinch distance is below the midpoint of its 10th and 90th percentiles (with hysteresis)."""
    lo, hi = np.percentile(width, [10, 90])
    thr = (lo + hi) / 2
    band = 0.15 * (hi - lo)
    closed = np.zeros(len(width), bool)
    state = width[0] < thr
    for t, w in enumerate(width):
        if state and w > thr + band:
            state = False
        elif not state and w < thr - band:
            state = True
        closed[t] = state
    return closed, float(thr)


def real_to_sim(bowl_real: np.ndarray, plate_real: np.ndarray, bowl_sim: np.ndarray, plate_sim: np.ndarray):
    """4x4 sim-from-table transform: real bowl centre -> sim bowl, bowl->plate direction matched (yaw only).
    Inputs are table-frame xy (real) and sim world xyz of each object's base (sim). Scale stays 1."""
    a = np.arctan2(*(plate_real - bowl_real)[::-1])
    b = np.arctan2(*(plate_sim - bowl_sim)[:2][::-1])
    Rz = R.from_euler("z", b - a).as_matrix()
    T = np.eye(4)
    T[:3, :3] = Rz
    T[:3, 3] = bowl_sim - Rz @ np.array([*bowl_real, 0.0])
    return T
