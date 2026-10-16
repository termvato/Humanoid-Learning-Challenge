# How I recorded the takes (Stray Scanner, iPhone with LiDAR)

The five raw takes are in the [raw-takes-v1 release](https://github.com/termvato/Humanoid-Learning-Challenge/releases/tag/raw-takes-v1).

## Set-up
- A desk with even light, a small white jug (the sim task uses a white mug) and a flat plate (23 cm across; the
  sim plate is 13.8 cm, which is why the demo multiplier centres the release on the plate).
- Jug and plate about 25-30 cm apart. Any direction works: the real-to-sim mapping turns the scene so the
  jug-to-plate direction matches the sim's.
- Only my right hand in view.

## Phone
- Held in my left hand, about 40-60 cm above the desk, looking down at about 45 degrees, landscape. Hand-held is
  fine: ARKit tracks the phone pose and the table plane stays within 0.3 cm over a take (spike S2).
- At the start of each take I kept only the desk, jug and plate in view for about 2 seconds; `tools/scene.py`
  finds the table and both objects in those frames.

## The motion (one take, about 8-15 s)
1. Hand comes in from the side, open.
2. Grasp the jug. takes 1-3: a whole-hand grab of the body. takes 4-5: a two-finger pinch of the rim, like a
   parallel gripper, with the back of my hand toward the phone. The pinch gives a much cleaner open/close signal.
3. Lift, carry over the plate, set down, release.
4. Move the hand away and hold still for 2 seconds.

## After
- Measured the jug-to-plate distance and the plate with a tape before moving them (spike S2 compares with LiDAR).
- Exported the Stray Scanner folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`,
  `imu.csv`) to `data/raw/takeNN/`.

## Run one take
```
.venv/bin/python tools/scene.py data/raw/take04 outputs/take04                 # table, plate, jug
.venv-hand/bin/python tools/extract_hand.py data/raw/take04 outputs/take04     # hand keypoints in the world frame
MUJOCO_GL=egl .venv/bin/python tools/replay.py data/raw/take04 outputs/take04 --task mug  # Panda replay + video
```
