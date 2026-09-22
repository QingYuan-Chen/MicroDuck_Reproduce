#!/usr/bin/env bash
# Run from repository root, after both bounded training arms complete.
set -euo pipefail
eval_root=logs/sampling_ab_20260920
eval_python=.venv/bin/python
for eval_seed in 20260921 20260922; do
  for eval_arm in baseline control specialist; do
    if [[ "$eval_arm" == baseline ]]; then
      eval_checkpoint=logs/rsl_rl/velocity/2026-09-20_00-47-08_velocity_env1024_long/model_3250.pt
    else
      eval_checkpoint="$eval_root/$eval_arm/model_3499.pt"
    fi
    # Seed-1 baseline was measured before training. Keep that original record.
    if [[ "$eval_seed" == 20260921 && "$eval_arm" == baseline ]]; then
      continue
    fi
    eval_name="${eval_arm}_scan_${eval_seed}"
    MJLAB_WARP_QUIET=1 MICRODUCK_WARM_START=0 "$eval_python" -u scripts/evaluate_walk_checkpoints.py \
      --checkpoint "$eval_checkpoint" --push none --suite scan --trials 16 \
      --seed "$eval_seed" --output "$eval_root/$eval_name.json" > "$eval_root/$eval_name.log" 2>&1
    printf 'Completed evaluation %s\n' "$eval_name"
  done
done
