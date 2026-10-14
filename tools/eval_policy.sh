#!/usr/bin/env bash
# Evaluate a policy on libero_90 #72 (stored init states, max 400 steps) and append the result to
# outputs/results.csv. Policy runs on the CPU so it can share the 8 GB GPU with training.
# Usage: tools/eval_policy.sh <name> <policy path or hub id> <episodes> [smolvla]
set -euo pipefail
name=$1; policy=$2; n=$3; kind=${4:-act}
out=outputs/eval/${name}_n${n}
extra=()
if [ "$kind" = smolvla ]; then
  extra=('--rename_map={"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}')
fi
MUJOCO_GL=egl .venv/bin/lerobot-eval --policy.path="$policy" --env.type=libero --env.task=libero_90 \
  '--env.task_ids=[72]' --env.observation_height=256 --env.observation_width=256 \
  --eval.n_episodes="$n" --eval.batch_size=2 --policy.device=cpu "${extra[@]}" --output_dir="$out" \
  > "$out.log" 2>&1
pc=$(grep -o "'pc_success': [0-9.]*" "$out.log" | head -1 | grep -o "[0-9.]*$")
[ -f outputs/results.csv ] || echo "name,policy,episodes,success_pct,date" > outputs/results.csv
echo "$name,$policy,$n,$pc,$(date -Iseconds)" >> outputs/results.csv
echo "$name $pc% of $n"
