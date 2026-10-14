# Ideas not yet decided (2026-10-14)

Time: about 55 h to the deadline (Fri 16 Oct, 23:59 BST). Hard constraint: data the user recorded must drive the approach.

## Shared first step (every option needs it)
Record a LIBERO-like tabletop task with the iPhone in Stray Scanner (RGB + LiDAR depth + ARKit poses), track the hand (MediaPipe or HaMeR), lift it to metric 3D with LiDAR depth, retarget wrist pose + grasp open/close to the Panda end effector, replay in LIBERO.
- This alone is the "hand-coded" floor the brief mentions. Hours, not days.
- Edge over a plain phone video: LiDAR gives metric hand and object positions, so no scale guessing.

## A. Post-train SmolVLA on own demos (the brief's example)
- Replayed trajectories that succeed in sim become LIBERO-format demos; fine-tune SmolVLA (450M) and report success rate vs the base model.
- Pro: exactly what they showed. Con: their own example, so least original. Fine-tuning likely needs Modal (8 GB is tight).

## B. One human demo -> many sim demos (object-centric augmentation, MimicGen style)
- Store the human trajectory relative to the object, not the table. Randomise object poses in sim, transform the trajectory, keep the ones that succeed: dozens of demos from a few phone recordings. Train a small policy (or SmolVLA) on them.
- Pro: creative, uses the LiDAR advantage, cheap compute. Con: works best for tasks with one object frame per phase.

## C. Bootstrap, then RL (residual PPO)
- Policy from A or B as the base; a small residual policy trained with PPO in sim corrects it. Report success before and after RL.
- Pro: matches the RL part of the role and the user's PPO experience. Con: most risk in 55 h; needs a working base first.

## D. World model
- Action-conditioned video prediction from own clips (hand trajectory as action).
- Pro: listed by them. Con: weakest fit for a mechanical engineer new to ML in 2 days; little to show in sim.

## E. Faster policy
- Quantise / distil SmolVLA for the laptop GPU, report latency vs success.
- Pro: measurable. Con: own data plays a small role, which the brief scores.
