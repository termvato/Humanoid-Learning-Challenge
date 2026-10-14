# Knowledge map

Start here. Every other doc is linked from this page. Update the status board at the end of each work unit.

## Status board (2026-10-14 evening)

| Area | State | Next action | Owner |
|---|---|---|---|
| Brief | saved: docs/challenge.md | | done |
| Research | done: docs/deep-research.md (verified), research.md (first pass) | user picks a plan | user |
| LIBERO on laptop (S1) | **works**: 42 steps/s at 256 px; mujoco 3.3.7 | | done |
| Task | libero_90 #72 put the white mug on the plate (D2) | | done |
| Scripted oracle | bowl 50/50 stored inits; mug grasp sweep 17/17; rim angle sweep: handle arc 88-127 deg | | done |
| Real takes | 5 takes (hotel, jug + plate); S2 drift PASS, S3 PASS (take04/05) | | done |
| Replay of real takes | 3/5 succeed with grasp anchor (D3); take01, take05 fail | take05 cause open | Claude |
| Demo multiplier | 58% yield at default layout (D4); 200 demos generating | train ACT | Claude |
| Policy | not trained | ACT locally, eval 100 episodes; zero-shot SmolVLA baseline | Claude |
| README | not started | Thu/Fri | Claude + user |

## Documents

| Doc | What it holds |
|---|---|
| [challenge.md](challenge.md) | the brief, verbatim |
| [decisions.md](decisions.md) | decisions with reasons; append-only |
| [spikes.md](spikes.md) | tests with pre-written rules and results |
| [ideas.md](ideas.md) | options not yet decided |
| [research.md](research.md) | first-pass findings, goals |
| [deep-research.md](deep-research.md) | verified research, ranked plans |
| [recording.md](recording.md) | how to record a take, and the commands |
