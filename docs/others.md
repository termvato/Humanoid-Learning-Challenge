# Other public submissions to the same brief (searched 2026-10-16)

Numbers are what each README reports; not checked by us.

| Repo | Task and sim | Capture | Policy | Reported result |
|---|---|---|---|---|
| andrealo20/phone-to-panda | LIBERO-Goal bowl on plate, 50 stored starts | phone, tablet checkerboard, MediaPipe, depth from table plane | SmolVLA from smolvla_base, 181 replays | 46% (70/150, 3 seeds); 13 teleop demos 92% |
| silvermew/hand-to-robot-cube | robosuite Panda, turn/carry a cube | stereo pair, ArUco on hand and cube | world model (cube angle) + SmolVLA | SmolVLA 84% on 50 new poses |
| Daveonwave/humanoid-challenge | libero_spatial bowl, one scene | iPhone + markers, scripted grasp | SmolVLA 2k steps | 0/1 rollout |
| palocga-bit/phone-to-panda | custom MuJoCo cube into box | phone, MediaPipe (58% of frames) | none | one scripted success |
| diw22/DWinters_Humanoid | G1 drumming to music | two cameras, MediaPipe, ArUco | OpenVLA-OFT + audio | 40/40 runs |

Overlaps with our findings:
- andrealo20: first SmolVLA 0%, "hovered over the bowl and pushed it", traced to wrist rotation in human replays
  (spread 0.67 rad vs 0.15 in teleop data). Fixes: wrist at rest, approach cut to last 10 cm, shorter carry, more
  balanced replays, 20k steps. Same symptom as our 400-demo SmolVLA runs.
- silvermew: random drift in demos gave 0% (our DART result); pauses in demos confuse the policy.

What we lack vs them: a teleop-demo baseline on the same task; a world model.
