"""LIBERO helpers: make the task env, read and set object poses, follow end-effector targets.

Frames: LIBERO world frame (z up). Quaternions are xyzw (robosuite obs and scipy), except MuJoCo qpos (wxyz).
"""

import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITE = "libero_90"


@dataclass(frozen=True)
class Task:
    task_id: int  # in libero_90
    pick: str  # object the hand picks up
    place: str  # object it goes on
    language: str


TASKS = {
    "bowl": Task(9, "akita_black_bowl_1", "plate_1", "put the black bowl on the plate"),
    "mug": Task(72, "porcelain_mug_1", "plate_1", "put the white mug on the plate"),
}
DEFAULT_TASK = "mug"
TASK_ID = TASKS[DEFAULT_TASK].task_id
POS_SCALE = 0.05  # m per unit action (OSC_POSE output_max)
ROT_SCALE = 0.5  # rad per unit action
GRIP_OPEN, GRIP_CLOSE = -1.0, 1.0


def make_env(suite: str = SUITE, task_id: int = TASK_ID, size: int = 256, init: int | None = 0, settle: int = 10):
    """Env plus the task; `init` picks one of the stored init states (None keeps the random reset).

    Steps `settle` no-op actions first, as LIBERO's evaluation does: the obs returned by reset and
    set_init_state are stale (bowl reported 7 cm above where it rests).
    """
    b = benchmark.get_benchmark_dict()[suite]()
    task = b.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size)
    env.seed(0)
    obs = env.reset()
    if init is not None:
        obs = env.set_init_state(b.get_task_init_states(task_id)[init])
    for _ in range(settle):
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN]))
    return env, task, obs


def eef_pose(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    return obs["robot0_eef_pos"].copy(), obs["robot0_eef_quat"].copy()


def obj_pose(obs: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    return obs[f"{name}_pos"].copy(), obs[f"{name}_quat"].copy()


def set_obj_pose(env, name: str, pos, quat_xyzw, settle: int = 10) -> dict:
    """Teleport a free object, then step no-op actions so it settles. Returns the new obs."""
    sim = env.env.sim
    joint = env.env.get_object(name).joints[-1]
    q = np.asarray(quat_xyzw)
    sim.data.set_joint_qpos(joint, np.concatenate([pos, [q[3], q[0], q[1], q[2]]]))
    sim.forward()
    obs = None
    for _ in range(settle):
        obs, *_ = env.step(np.array([0, 0, 0, 0, 0, 0, GRIP_OPEN]))
    return obs


def pose_to_action(cur_pos, cur_quat, tgt_pos, tgt_quat, grip: float) -> np.ndarray:
    """Delta OSC action that moves the current pose toward the target (clipped to one step)."""
    dpos = (np.asarray(tgt_pos) - cur_pos) / POS_SCALE
    drot = (R.from_quat(tgt_quat) * R.from_quat(cur_quat).inv()).as_rotvec() / ROT_SCALE
    return np.concatenate([np.clip(dpos, -1, 1), np.clip(drot, -1, 1), [grip]])


def follow(env, obs, targets, max_steps_per_target: int = 60, tol: float = 0.005, frames: list | None = None):
    """Drive the gripper through (pos, quat, grip, hold_steps) targets. Returns (obs, success, steps)."""
    steps, success = 0, False
    for pos, quat, grip, hold in targets:
        for k in range(max_steps_per_target):
            p, q = eef_pose(obs)
            if np.linalg.norm(np.asarray(pos) - p) < tol and k >= hold:
                break
            obs, _, _done, _ = env.step(pose_to_action(p, q, pos, quat, grip))
            steps += 1
            if frames is not None:
                frames.append(obs["agentview_image"][::-1])
            success = success or bool(env.env._check_success())
    return obs, success, steps
