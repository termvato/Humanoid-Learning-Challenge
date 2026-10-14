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
