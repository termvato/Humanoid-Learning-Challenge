# Spikes: cheap tests before committing (2026-10-14)

Each spike has a pass/fail rule written BEFORE running it, so the result decides, not our hopes.

| ID | Question | Test | Decision rule | Cost |
|---|---|---|---|---|
| S1 | Does LIBERO run on the laptop? | Install LeRobot with the libero and smolvla extras (uv, Python 3.12), load one libero_spatial task, step 100 random actions with MUJOCO_GL=egl, render agentview 256x256 | Steps and renders >= 20 fps -> local sim is the dev loop. Install fails after 1.5 h of effort or < 5 fps -> sim on Modal | laptop |
| S2 | Are the phone's poses and depth good enough? | One Stray take of bowl and plate on a table: table points from the first and last frame in world frame; bowl-to-plate centre distance from LiDAR vs tape measure | Table planes agree within 1 cm (median point-to-plane) AND bowl-plate distance within 2 cm of the tape -> geometry trusted. Else fix pose convention / intrinsics scale before any hand code | 20 min |
| S3 | Does WiLoR + LiDAR give a usable gripper path? | Same take: per frame WiLoR keypoints, depth-corrected, in world frame; gripper point = thumb-index tip midpoint | Hand found in >= 90% of frames while moving AND the gripper point is within 3 cm of the bowl rim at the grasp moment (closest approach) -> retargeting goes ahead. Else try MediaPipe or fingertip depth from the mask | laptop |

## Results
- **S1 PASS** (2026-10-14): installs on Python 3.12 (lerobot 0.6.1, robosuite 1.4.0, mujoco 3.8.1); one env 42 steps/s at 256x256, 107 at 128x128, both cameras rendered. Laptop is the dev loop. Open: research says pin mujoco<3.4 (stored init states change from 3.4). `outputs/s1_libero/`
- **Oracle** (2026-10-14): scripted pick-and-place with true object poses (rim grasp 5 cm from bowl centre along y, 4 cm above its base) succeeds on 50/50 stored init states and 34/50 wider placements (+-10 cm). Gotcha: obs right after reset/set_init_state is stale (bowl 7 cm too high); step 10 no-ops first. `outputs/oracle/`
- **Hand tracker** (2026-10-14): WiLoR-mini on the 4070, fp16: 38 ms/frame (2 hands), 1.5 GB peak. Outputs 21 keypoints 2D + 3D, MANO vertices. Separate env `.venv-hand` (chumpy needs `--no-build-isolation-package chumpy`, plus `dill`).

## Real takes, 2026-10-14 (hotel desk, real plate, white jug)

Five takes (60 fps, 1920x1440). take01-03: whole-hand grab of the jug body. take04-05: two-finger rim pinch.

- S2 table drift: 0.14-0.30 cm on all five. PASS (rule < 1 cm). Tape-measure check not done.
- Object finding needed two height bands (plate < 3.5 cm, jug > 3 cm, table cells under the jug masked) and a circle
  fit to the visible arc: the jug's base merged into the plate, and the shelf was taken for the jug. After the fix all
  five agree: plate 23 cm, jug 9-10 cm tall, radius 3-4 cm.
- S3 hand found in 100% of in-view frames on take04/05 (rule >= 90%). PASS. Pinch point vs rim at the grasp
  1.4-1.7 cm radial (rule < 3 cm). PASS.
- Raw retargeted replay: 0/2 (take04/05). The sim grasp landed 6-8 cm from the mug centre (rim at 4 cm).
- Grasp anchor (one constant shift of the whole track so the grasp lands on the sim rim, same angle, oracle height):
  shifts of 2-5 cm, mostly downwards. Replay success 3/5: take02, take03, take04. take01 (whole-hand grab) and
  take05 fail.
- Rim angle sweep (scripted radial rim pinch, rule: failures cluster in one arc containing take05's 84 deg):
  30 deg steps fail at 90 and 120 only; 5 deg steps fail 90-125 (the handle). The anchor now moves grasps out of
  88-127 deg. take05 grasps at 84 deg and still fails, so the handle is not the whole story there; open.
- Hand detection by LiDAR on the real jug handle: unreliable (handle mostly hidden), not used.

## Policy v1: ACT on 200 demos from 3 real takes (2026-10-15)

Data: `data/demos/real3_r04` (200 kept of 448 rollouts, 60%; sources take02/03/04; placements +-4 cm around the
LIBERO default layout; release centred on the plate). ACT, chunk 50, 20 actions per call, batch 32, 20k steps
(~17 epochs), RTX 4070, ~75 min. Eval: lerobot-eval on libero_90 #72 stored init states, max 400 steps.

| checkpoint | episodes | success |
|---|---|---|
| 5k | 10 | 0% |
| 10k | 20 | 10% |
| 15k | 50 | 14% |
| 20k | 50 | 20% |

Failures (10k videos): grasps that miss the rim, arm drifting off the mug, mug knocked over, plate flipped once.
Next: DART-style action noise in the demos (noise 0.2: yield unchanged, 24/39).

## Night of 2026-10-15: baselines, noise, chunk length, Modal

All on libero_90 #72 stored init states, 100 episodes unless noted (95% CI about +-8 points at 20%).

- Zero-shot `lerobot/smolvla_libero` (LIBERO-trained VLA, no own data): 2%.
- ACT on clean demos (200, 3 takes): 14% at 20k, 20% at 30k, 16% at 40k steps. Plateau.
- ACT on DART-noise demos (noise 0.2): 8% at 20k, 16% at 30k. Noise did not help.
- **Action chunk length at eval** (same clean 30k checkpoint, no retraining): 20 actions per call 20%,
  10 actions 32%, 5 actions 36%. The policy needs to re-plan; 1 s blind is too long for the grasp.
- LIBERO init states: mug and plate each span ~4.7 cm, mug yaw fixed at -90 deg. Mean positions are
  2.6 cm from init state 0, so v3 demos sample +-3.5 cm around the mean (yield 66%, 400 demos).
- Oracle source (scripted rim pinch, true mug pose) through the same multiplier: yield 88% (vs 66% phone).
- Modal: `modal run --detach` jobs were cancelled when the laptop lost its connection; spawned calls on a
  deployed app survive. Then the workspace spend limit stopped all jobs mid-run; only checkpoints already
  committed survived (SmolVLA 5k x2, ACT v3 10k).
- DART noise hurts both learners: ACT (5 actions/call) 36% clean vs 14% noisy; SmolVLA 5k (10 actions/call,
  50 episodes) 48% clean vs 32% noisy. ACT temporal ensembling 36%, same as 5 actions/call.

## 2026-10-16 morning: oracle vs human source, same recipe

Question: is the ~36% ceiling of ACT the learner or the human-derived data? Same multiplier (around the init mean,
+-3.5 cm, clean), same ACT recipe (chunk 50, batch 32, 30k steps), eval 5 actions per call, 100 episodes.

| source | multiplier yield | ACT success |
|---|---|---|
| scripted oracle rim pinch | 97% (400 demos) | **92%** |
| 3 phone takes | 87% (400 demos) | 34% |
| 3 phone takes, pauses removed (`tools/retime.py`) | 88% (400 demos) | **64%** |
| 3 phone takes, older 200-demo set (r04) | 60% | 36% |

Clean v3 yields are higher than the noisy v3 runs (97/87% vs 88/66%): the noise itself caused failed rollouts.
The learner can reach 92% on this task with this amount of data, so the gap is in the demos.
Modal still refuses new calls (workspace spend limit), so all of this runs on the laptop.

### Spike: retimed human tracks (rule written before running)
The phone tracks are still 43-65% of the time (oracle 20%); take04 goes 275 -> 149 steps without its pauses.
Same multiplier and ACT recipe as act_mean_30k, sources retimed with `tools/retime.py` (2 mm path step, 12-frame
dwell after grip changes). Rule: if ACT success (100 episodes, 5 actions per call) beats act_mean_30k by 10+
points, retiming becomes part of the pipeline; within 10 points, it is reported as no effect.

Result: 64% vs 34% (100 episodes each). PASS (+30 points): retiming is part of the pipeline (D5).

### SmolVLA 20k round on the 400-demo sets (Modal, 2026-10-16)
- 5k checkpoints: normal demos 0%, retimed demos 0% (100 episodes each). The earlier 5k run on the 200-demo set
  scored 46%, and re-evaluating it today gives 2/10, so the eval is not broken. Configs differ only in the dataset;
  weights have no NaNs and moved from the base as much as the working run.
- Behaviour: reaches the mug, presses on it, then the wrist slowly twists and tilts; the mug is never lifted.
- 5k steps at batch 16 is 1.2 epochs of the 400-demo sets vs 2.1 of the 200-demo set. 10k/15k/20k pending.

### Spike: still wrist (rule written before running)
Human-derived demos turn the wrist ~6x more than scripted ones; andrealo20 (docs/others.md) traced the same
"hover and push" SmolVLA failure to this. Sources: retimed takes with the gripper held at its grasp orientation
(`tools/retime.py --hold-wrist`), 400 demos, SmolVLA 10k and ACT 30k on Modal. Rule: a fix if SmolVLA beats 46%
or ACT beats 64% by 10+ points (100 episodes); otherwise reported as no effect.

### Spike: action-conditioned world model (rule written before running)
`infra/modal_world.py`: SD-VAE latents (128 px, both cameras), FiLM conv net predicts 0.2 s ahead from two frames
and 4 actions, trained unrolled on retimed, as-recorded and scripted demos; 1 in 20 episodes held out.
Rules: (1) at 1 s ahead, its latent error is below "nothing moves" (copy last frame) on held-out episodes;
(2) imagining a held-out demo's own actions ends with the mug on the plate (classifier on real frames) in 70%+
of episodes, and the same actions with the gripper never closing in 30% or fewer. Both must hold to call it a
world model that understands grasping; otherwise reported as what it got right and wrong.
Result wm1 (15k steps, H100, 24 min; 1200 episodes, 60 held out):
- Rule 1 PASS. Latent error vs "nothing moves": 0.029 vs 0.228 at 0.2 s, 0.032 vs 0.410 at 1 s, 0.038 vs 0.489 at 2 s.
- Rule 2 FAIL as written: imagined own actions end "on plate" 48% (needed 70%), gripper never closing 18% (pass).
  The judge is the bottleneck: the classifier calls only 52% of the real final frames "on plate". Cause found
  after the result: the 10 steps before the release (mug already on the plate, still held) were labelled "not on
  plate" and look the same. Rerun of the classifier only, those frames left out (wm1_cls2), reported separately.
- Videos: imagined rollouts with the demo's actions follow the real ones; with the gripper never closing the
  image smears where the carry should be (never seen in training).
Result wm1_cls2 (same world model, classifier retrained with the 20 pre-release steps left out): judge calls 68%
of real final frames "on plate" (was 52%); imagined own actions 70%, gripper never closing 47%. Rule 2 still FAIL:
the gap between grasping and not grasping is small. The world model predicts motion well (rule 1) but does not
reliably imagine that an ungrasped mug stays behind; it has never seen an open gripper during a carry.

### Spike: world model with no-grasp episodes (rule written before running)
wm1 never saw an open gripper during a carry. Add 300 sim episodes that follow the retimed tracks with the gripper
never closing (`--keep-failures`; the mug stays put), retrain wm1's recipe (wm2), same classifier recipe as
wm1_cls2. Counterfactuals on held-out normal demos. Rules: own actions end "on plate" 70%+; gripper never closing
30% or less (now in distribution); gripper opening halfway through the carry 30% or less (in no training episode:
the real test of "the mug moves only while held").

### Spike: release early (rule written before running)
ACT 64% failures (videos): carries the mug over the plate and hovers without opening. Demos from the takes open
only after pressing the mug onto the plate (eef z 0.531 at release, 3 cm descent in the last 15 steps; take04 even
opens after lifting again) while scripted demos open right after a fast descent and drop the mug ~2 cm (z 0.586).
Variant: retimed takes with the gripper opening 2 cm above the set-down point (`tools/retime.py --release-early`),
400 demos, same ACT recipe on Modal, 100 episodes at 5 actions per call. Rule: a fix if it beats 64% by 10+ points.
Result wm2 (wm1 recipe + 300 no-grasp episodes; 1500 episodes, 60 held-out demos): latent error 1 s ahead 0.034
vs 0.415 for copying the last frame. Imagined endings "on plate": own actions 68% (rule 70%+: FAIL by 2 points; the
classifier gives real final frames 68%), gripper never closing 0% (PASS, was 47%), gripper opening halfway
through the carry 28% (PASS; no training episode does this). Adding failure data taught it that the mug moves only
while held, including for a release it never saw.

### Actions per call on the retimed ACT (64% at 5)
Same checkpoint, 100 episodes: 5 actions per call 64%, 10 actions 36%, temporal ensembling (1 action, coeff 0.01)
8%. On the earlier 200-demo ACT the same settings gave 36% / 32% / 36%. The retimed policy depends strongly on
re-planning every 5 steps; since 5 was chosen on the test states, 64% is the best of three settings. Seed repeats
running.
