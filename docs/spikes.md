# Spikes: cheap tests before committing (2026-10-14)

Each spike has a pass/fail rule written BEFORE running it, so the result decides, not our hopes.

| ID | Question | Test | Decision rule | Cost |
|---|---|---|---|---|
| S1 | Does LIBERO run on the laptop? | Install LeRobot with the libero and smolvla extras (uv, Python 3.12), load one libero_spatial task, step 100 random actions with MUJOCO_GL=egl, render agentview 256x256 | Steps and renders >= 20 fps -> local sim is the dev loop. Install fails after 1.5 h of effort or < 5 fps -> sim on Modal | laptop |

## Results
- **S1 PASS** (2026-10-14): installs on Python 3.12 (lerobot 0.6.1, robosuite 1.4.0, mujoco 3.8.1); one env 42 steps/s at 256x256, 107 at 128x128, both cameras rendered. Laptop is the dev loop. Open: research says pin mujoco<3.4 (stored init states change from 3.4). `outputs/s1_libero/`
