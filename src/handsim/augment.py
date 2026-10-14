"""One retargeted demo -> many: move the pick and place objects, carry the path along (MimicGen-style,
object-centric).

The approach and grasp ride with the pick object, the release rides with the place object, and the carry in
between blends from one shift to the other. Rollouts are kept only if the sim reports success.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

from handsim.sim import GRIP_CLOSE, Task, eef_pose, obj_pose, pose_to_action, set_obj_pose


def shift_profile(grip: np.ndarray) -> np.ndarray:
    """Per step weight w: offset = (1 - w) * pick shift + w * place shift. 0 until the grasp, 1 from the release."""
    closed = np.flatnonzero(grip == GRIP_CLOSE)
    w = np.zeros(len(grip))
    if len(closed) == 0:
        return w
    g = closed[0]
    after = np.flatnonzero((np.arange(len(grip)) > g) & (grip != GRIP_CLOSE))
    r = after[0] if len(after) else len(grip) - 1
    w[g:r] = np.linspace(0, 1, r - g, endpoint=False)
    w[r:] = 1
    return w


def warp(eef: np.ndarray, grip: np.ndarray, d_pick: np.ndarray, d_place: np.ndarray) -> np.ndarray:
    w = shift_profile(grip)[:, None]
    return eef + (1 - w) * d_pick + w * d_place


def quat2axisangle(q: np.ndarray) -> np.ndarray:
    """xyzw -> axis-angle, as LeRobot's LIBERO processor does (angle in [0, pi])."""
    q = q * (1 if q[3] >= 0 else -1)
    return R.from_quat(q).as_rotvec()


def rollout(env, obs, eef, quat, grip, record: bool, extra_steps: int = 10):
    """Run the target sequence. Returns (success, frames) with frames as LeRobot LIBERO features."""
    frames, success, after = [], False, 0

    def add(o, action):
        if record:
            frames.append(
                {
                    "observation.images.image": o["agentview_image"][::-1, ::-1].copy(),
                    "observation.images.image2": o["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
                    "observation.state": np.r_[
                        o["robot0_eef_pos"], quat2axisangle(o["robot0_eef_quat"]), o["robot0_gripper_qpos"]
                    ].astype(np.float32),
                    "action": action.astype(np.float32),
                }
            )

    # approach the first target with recorded actions too, so the policy learns the start
    for _ in range(80):
        p, q = eef_pose(obs)
        if np.linalg.norm(eef[0] - p) < 0.005:
            break
        a = pose_to_action(p, q, eef[0], quat[0], grip[0])
        add(obs, a)
        obs, *_ = env.step(a)
    for i in range(len(eef)):
        p, q = eef_pose(obs)
        a = pose_to_action(p, q, eef[i], quat[i], grip[i])
        add(obs, a)
        obs, *_ = env.step(a)
        if env.env._check_success():
            success = True
        if success:
            after += 1
            if after > extra_steps:
                break
    return success, frames


def place(env, task: Task, pick_xy: np.ndarray, place_xy: np.ndarray):
    """Move both objects (xy), keeping their heights and orientations, then let them settle."""
    obs = env.env._get_observations()
    a, aq = obj_pose(obs, task.pick)
    b, bq = obj_pose(obs, task.place)
    set_obj_pose(env, task.place, np.r_[place_xy, b[2] + 0.005], bq, settle=0)
    return set_obj_pose(env, task.pick, np.r_[pick_xy, a[2] + 0.005], aq)
