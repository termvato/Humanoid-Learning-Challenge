# Phone video to a Panda in LIBERO: put the mug on the plate

I recorded myself moving a white jug onto a plate with an iPhone (LiDAR depth plus phone pose). This repo turns
that recording into robot demonstrations for LIBERO task `libero_90 #72` ("put the white mug on the plate"),
multiplies them in simulation, and trains policies (ACT and the SmolVLA VLA) on nothing but that data.

![phone vs sim](docs/media/take04_phone_vs_sim.gif)

*Left: my hand, recorded in a hotel room. Right: the Panda replaying the retargeted motion. LIBERO's own success check fires.*

## Results

All success rates are LIBERO's own task check on the 50 stored start states of `libero_90 #72`, 400 steps max,
evaluated with `lerobot-eval`. 100 episodes per row unless noted; the 95% interval at 20% is about +-8 points.

| What | Data | Success |
|---|---|---|
| Open-loop replay of each phone take (own layout) | 5 takes | 3/5 takes |
| Demo multiplier yield (replay at new object positions, kept only if the sim says success) | 3 takes | 60-66% of rollouts |
| SmolVLA `lerobot/smolvla_libero`, zero-shot (no own data) | none | 2% |
| ACT, 30k steps, 20 actions per call | 200 demos from 3 takes | 20% |
| same ACT checkpoint, 5 actions per call | 200 demos from 3 takes | 36% |
| same ACT checkpoint, temporal ensembling (1 action per call) | 200 demos from 3 takes | 36% |
| ACT, 30k steps, noisy demos (DART), 20 actions per call | 200 demos from 3 takes | 16% |
| **SmolVLA fine-tuned from `smolvla_libero`, 5k steps, 10 actions per call** | 200 demos from 3 takes | **48%** (50 episodes) |

![ACT training curve](docs/media/act_training_curve.png)

## How it works

```
iPhone (Stray Scanner: RGB 1920x1440 @ 60 fps, LiDAR depth, ARKit poses)
  -> tools/scene.py        table plane, plate and jug from LiDAR (metric, gravity-aligned)
  -> tools/extract_hand.py WiLoR hand keypoints per frame, placed in 3D with the phone intrinsics + LiDAR depth
  -> tools/replay.py       thumb/index pinch -> Panda gripper pose and open/close; real -> sim transform;
                           grasp anchor; open-loop replay in LIBERO (OSC_POSE, 20 Hz)
  -> tools/generate_demos.py  MimicGen-style: move mug and plate, warp the path with them, keep sim successes
                              -> LeRobotDataset (same format as the public LIBERO datasets)
  -> lerobot-train (ACT locally; ACT and SmolVLA on Modal, infra/modal_train.py)
  -> tools/eval_policy.sh  lerobot-eval on libero_90 #72, appends to outputs/results.csv
```

![phone frames](docs/media/take04_frames.jpg)

### Design choices, and why

- **Stay on a stock LIBERO task.** `libero_90 #72` uses a white mug and a plate, which I could reproduce with
  a hotel jug and plate. No custom scene, so the success check and start states are LIBERO's own.
- **LiDAR, not monocular depth.** The phone gives metric depth and its own pose for free. The table plane drifts
  0.14-0.30 cm over a take (test S2); the plate measures 23 cm, which matches a tape measure.
- **Hand to gripper.** WiLoR gives 21 hand keypoints. The gripper frame comes from the knuckles (stable while
  pinching), the gripper point is the thumb/index tip midpoint, and open/close is the tip distance with hysteresis.
  The hand's 3D position is solved from the 2D keypoints with the phone's real intrinsics, then slid along the
  viewing ray to agree with LiDAR. The hand is found in 100% of frames where it is in view (test S3).
- **Grasp anchor.** Raw retargeting failed 2/2: the grasp landed 6-8 cm from the mug centre (rim at 4 cm). Hand
  tracking is off by a few cm, mostly as a constant bias, and a rim pinch needs about 1 cm. At the moment the
  fingers close we know they are on the rim, so the whole track is shifted by one vector to put that point on
  the sim mug's rim (2-5 cm shifts). The human's carry and place keep their shape.
- **Handle arc.** A scripted sweep around the sim mug shows pinches between 90 and 125 deg fail: that is the
  handle. Grasps landing there are moved to the arc's edge.

  ![rim sweep](docs/media/rim_sweep.png)
- **Demo multiplier.** One human take is one layout. Each rollout moves the mug and plate (around the mean of
  LIBERO's start states), shifts the approach with the mug and the release with the plate, and is kept only if the
  sim reports success. The release is centred on the plate: my plate is 23 cm, LIBERO's 13.8 cm, so my
  off-centre set-downs missed the sim plate (yield 40% -> 60%).
- **DART noise.** Clean replays never show recovering from an error, and ACT's failures were missed grasps it
  could not recover from. The noisy variant perturbs executed actions and records the clean corrective action.
  It did not help ACT here (16% vs 20%, inside the noise).
- **Two policy families.** ACT (small, trains on the laptop in about an hour) and SmolVLA (a 450M VLA, fine-tuned
  from the LIBERO checkpoint on a Modal L4) on the same data.

## What did not work

- **Whole-hand grabs.** Takes 1-3 grab the jug body with the whole hand; the thumb/index distance is a weak
  open/close signal then. take01 fails replay. I re-recorded with a two-finger rim pinch (take04, take05).
- **take05** fails replay even after the anchor, and its grasp angle works in the scripted sweep. Cause not found.
- **Finding the jug's handle in LiDAR**, to align real and sim handles, was unreliable: the handle is mostly hidden
  behind the jug. Replaced by the handle arc rule above.
- **Plain object finding** merged the jug into the plate and once took a shelf for the jug. Fixed with two
  height bands and a circle fit to the visible arc.
- **ACT plateaus around 15-20%** on clean demos: 14% at 20k, 20% at 30k, 16% at 40k steps (100 episodes each).
  Failures are mostly grasps that miss the rim, drift, or knock the mug over.
- **Engineering:** 24 parallel sim workers ran the laptop out of RAM and GPU memory (8 workers now), and
  `modal run --detach` jobs were cancelled when the laptop lost its connection (jobs are now spawned on a
  deployed app).

## How to run

```bash
uv venv .venv && uv pip install --python .venv/bin/python "lerobot[libero,smolvla]==0.6.1" "mujoco<3.4" -e .
# hand tracker in its own env (torch cu126, wilor-mini): see docs/recording.md

# 1. one take (Stray Scanner export) -> scene, hand, replay
.venv/bin/python tools/scene.py data/raw/take04 outputs/take04
.venv-hand/bin/python tools/extract_hand.py data/raw/take04 outputs/take04
MUJOCO_GL=egl .venv/bin/python tools/replay.py data/raw/take04 outputs/take04 --task mug

# 2. demos from the takes that replay successfully
MUJOCO_GL=egl .venv/bin/python tools/generate_demos.py outputs/take0{2,3,4}/traj_sim.npz \
    --n 200 --radius 0.035 --around init_mean --noise 0.2 --out data/demos/real3

# 3. train and evaluate
.venv/bin/lerobot-train --dataset.repo_id=local/real3 --dataset.root=data/demos/real3 --policy.type=act \
    --policy.chunk_size=50 --policy.n_action_steps=20 --batch_size=32 --steps=30000 --policy.push_to_hub=false
tools/eval_policy.sh act outputs/train/<run>/checkpoints/030000/pretrained_model 100
```

Tests: `.venv/bin/python -m pytest` (geometry, augmentation, Stray loader).

## Repo map

| Path | What |
|---|---|
| `src/handsim/` | geometry (plane, objects, hand to gripper, real to sim, grasp anchor), sim helpers, demo augmentation |
| `tools/` | the pipeline steps above, figures, eval |
| `infra/` | Modal training app and launcher |
| `spikes/` | tests with rules written before running (LIBERO speed, oracle grasps, rim sweep) |
| `docs/` | brief, decisions (D1-D4), spike results, research notes |
