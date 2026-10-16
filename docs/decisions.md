# Design decisions

## D1. Working rules (2026-10-14)
- I decide the design and review each part; Claude Code (an AI coding agent) writes and runs the code. Direction
  choices are settled by small tests with pass/fail rules written before running (`spikes.md`), and every run
  writes its results to files.
- Why: the same rules worked on my Spatial AI challenge, and they keep the results honest.

## D2. Task: libero_90 #72 "put the white mug on the plate" (2026-10-14)
- I recorded with a white jug and a real plate in a hotel room. This stays inside the brief: a stock LIBERO task.
- Why: closest match to what I had at hand; the bowl task #9 is kept as a fallback in `TASKS`.

## D3. Grasp anchor: one constant shift of the whole retargeted track (2026-10-14)
- At the first closed frame my fingertips are on the rim, so I shift the track to put that point on the sim
  mug's rim (same angle around it, oracle grasp height). Measured shifts 2-5 cm.
- Why: raw retargeting failed 2/2; hand tracking is a few cm off as a mostly constant bias, and a rim pinch needs
  ~1 cm. A constant shift keeps my carry and place shape.
- Grasps inside the sim mug's handle arc (88-127 deg, measured by a rim sweep) move to the arc's end.

## D4. Demos sampled around LIBERO's default layout, release centred on the plate (2026-10-14)
- The demo multiplier samples object positions around the task's default positions (what LIBERO evaluates), and
  shifts the release so the held mug's centre lands on the plate centre.
- Why: my plate is 23 cm, the sim plate 13.8 cm; my off-centre set-downs miss the sim plate. Yield at the
  default layout went from ~40% to 58% (24 rollouts).

## D5. Remove my pauses from the source tracks before multiplying (2026-10-16)
- `tools/retime.py` keeps a frame only after 2 mm of hand motion, plus 12 frames after each gripper change.
- Why: my hand was still 43-65% of the time in the takes. ACT on the same 400-demo recipe went from 34% to 64%
  with pauses removed (spike rule: +10 points). Doubling the demos (200 -> 400) without retiming did not help
  (36% -> 34%).
- Update (2026-10-16, evening): two more training seeds of the retimed recipe gave 36% and 24% (mean of three 41%),
  so the 34% -> 64% gain is mostly seed luck. Retiming stays in the pipeline (it does not hurt, and episodes are
  shorter), but it is not a proven improvement.
