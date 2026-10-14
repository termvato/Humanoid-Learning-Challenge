# Design decisions

## D1. Same working rules as spatial-project (2026-10-14)
- User decides, Claude builds; spikes with pre-written rules settle direction choices; results to files.
- Why: the user asked for it; the rules worked on the Spatial AI challenge.

## D2. Task: libero_90 #72 "put the white mug on the plate" (2026-10-14)
- The user recorded with a white jug and a real plate (hotel). Stays inside the brief: a stock LIBERO task.
- Why: closest household match; bowl task #9 kept as a fallback in `TASKS`.

## D3. Grasp anchor: one constant shift of the whole retargeted track (2026-10-14)
- At the first closed frame the fingertips are on the rim, so the track is shifted to put that point on the sim
  mug's rim (same angle around it, oracle grasp height). Measured shifts 2-5 cm.
- Why: raw retargeting failed 2/2; hand tracking is a few cm off as a mostly constant bias, a rim pinch needs ~1 cm.
  A constant shift keeps the human's carry and place shape.
- Grasps inside the sim mug's handle arc (88-127 deg, measured by a rim sweep) move to the arc's end.

## D4. Demos sampled around LIBERO's default layout, release centred on the plate (2026-10-14)
- The demo multiplier samples object positions around the task's default positions (what LIBERO evaluates), and
  shifts the release so the held mug's centre lands on the plate centre.
- Why: the real plate is 23 cm, the sim plate 13.8 cm; off-centre human set-downs miss the sim plate. Yield at the
  default layout went from ~40% to 58% (24 rollouts).
