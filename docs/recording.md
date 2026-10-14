# How to record a take (Stray Scanner, iPhone with LiDAR)

## Set up
- Table clear, good even light, no strong sun patches.
- A small bowl (sim bowl: 11 cm across, 5 cm tall) and a flat plate (sim: 14 cm across). Similar sizes help; exact match not needed.
- Bowl and plate about 25-30 cm apart (the sim has 28 cm), plate to the right of the bowl as you face the table is fine; any direction works, the mapping turns it.
- Use your **right hand**. No other hands in view.

## Phone
- Hold the phone in your **left hand**, about 40-60 cm above the table, looking down at about 45 degrees, **landscape**. Keep it as still as you can (it does not need to be perfectly still; ARKit tracks it).
- Stray Scanner: start recording, then wait **2 seconds** with only the table, bowl and plate in view (this is how the code finds them).

## The motion (one take, about 8-10 s)
1. Hand comes in from the side, open.
2. **Pinch the bowl's rim** between thumb and index finger, on the side away from the plate (like a robot gripper: two fingers, not a whole-hand grab). Keep the back of your hand toward the phone.
3. Lift about 10 cm, move over the plate, lower, release.
4. Move the hand away, hold 2 seconds, stop.
- Move a bit slower than natural. Keep thumb and index tips visible as much as possible.

## After
- Measure the bowl-to-plate centre distance with a tape before you move them (test S2 compares it with LiDAR).
- Export from Stray Scanner (Files / AirDrop) to the laptop: the folder with `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`.
- Put it in `data/raw/take01/`.

## Run
```
.venv/bin/python tools/scene.py data/raw/take01 outputs/take01           # S2: table, bowl, plate
.venv-hand/bin/python tools/extract_hand.py data/raw/take01 outputs/take01 # hand keypoints in world frame
.venv/bin/python tools/replay.py data/raw/take01 outputs/take01           # Panda replay + side by side video
```
Record 3-5 takes once take01 works (different bowl/plate positions), for more source demos.
