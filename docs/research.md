# Research: what the challenge needs (2026-10-14)

## The brief in one line
Data the user records with a phone must drive a Panda arm in a simple sim (LIBERO suggested). Scored on: creativity (own data must matter), policy success in sim and/or world model quality, simple code and clear results "without AI slop". Repo + README by Fri 16 Oct, 23:59 BST.

## Their example
Egocentric hand video (left) -> Panda in LIBERO driven by a SmolVLA-based policy (right). They list as options: post-train on own data, creative retargeting to other embodiments, RL on a bootstrapped policy, world model prediction, faster policy.

## Findings
- **LIBERO**: MuJoCo/robosuite tabletop tasks with a Panda (e.g. "pick up the black bowl and place it on the plate"). LeRobot installs it with `pip install -e ".[libero]"` (Linux, `MUJOCO_GL=egl`) and evaluates with `lerobot-eval --env.type=libero --env.task=libero_object`. Original LIBERO pins robosuite 1.4.0 / mujoco 3.2.3 / old numpy; use LeRobot's packaged version, not the original repo. [lerobot docs](https://huggingface.co/docs/lerobot/libero)
- **SmolVLA**: 450M VLA from LeRobot, reported ~87% average on LIBERO (secondary source, not checked). A base checkpoint already fine-tuned on LIBERO exists (`lerobot/smolvla_libero`). A community run fine-tuned it on LIBERO-Object on one T4 in ~5 h (10k steps, batch 8, vision and language frozen), so our 8 GB laptop or a cheap Modal GPU is enough. [HF model](https://huggingface.co/KotaroT1/smolvla_libero_object_ft)
- **Hand to gripper retargeting, the standard recipe**: HaMeR per frame for hand pose; its 3D position from one RGB image is unreliable, so align with depth (Phantom: depth + ICP). Gripper position = midpoint of thumb and index tips; gripper width from their distance; wrist orientation is noisy and often needs smoothing or optimisation. Works for pinch grasps, fails for whole-hand grasps. [Phantom](https://arxiv.org/pdf/2503.00779), [RoboWM-Bench](https://arxiv.org/pdf/2604.19092), [HumanEgo](https://arxiv.org/pdf/2605.24934), [R+X](https://arxiv.org/pdf/2407.12957)
- **Field direction in 2026**: egocentric human video as robot training data is hot (HuRo, Ego2Robot, EgoEngine with residual RL after retargeting). So "phone video -> robot demos" is on trend; the differentiator must be in how well and how creatively. [HuRo](https://arxiv.org/pdf/2609.10706), [Ego2Robot](https://arxiv.org/pdf/2608.02580), [EgoEngine](https://arxiv.org/pdf/2606.12604)

## Our assets
- iPhone with Stray Scanner: RGB + LiDAR depth + ARKit poses. This gives the metric 3D hand position that HaMeR alone gets wrong, without ICP tricks. Loader and pose conventions already solved in `~/spatial-project`.
- User's PPO experience (`~/isaac-ros-pendulum`); laptop RTX 4070 8 GB; Modal account.

## Key problems to solve
1. **Embodiment gap**: phone pixels show a hand, LIBERO's camera shows a Panda. Simplest bridge: own data supplies *trajectories* (retargeted end-effector paths), replayed in sim; the policy trains on sim renders of those replays. Then own data drives the robot without the policy ever seeing a hand.
2. **Scene match**: real objects and layout must resemble one LIBERO task (bowl, plate, mug are easy to find at home), and the real table frame must map to the sim table frame.
3. **Few demos**: a handful of recordings is too little for a policy. Fix: object-centric augmentation (express the path relative to the object, re-target to randomised sim object poses, keep successes).
4. **Time**: ~55 h.

## Goals
- **G0 (must, Thu midday)**: own recordings -> hand track (LiDAR metric) -> Panda end-effector path -> open-loop replay in a matching LIBERO scene, with a side-by-side video like theirs. Meets the hard constraint even if everything else fails.
- **G1 (core, Thu night)**: a policy trained from own data with a measured success rate (N randomised episodes) vs a baseline (e.g. base SmolVLA or plain replay).
- **G2 (stretch, Fri)**: one creative step on top, chosen by result: residual RL to lift success, or retargeting to a different embodiment.
- **G3 (Fri, 20:00 buffer)**: README with results, what failed, videos; public repo; submit.
