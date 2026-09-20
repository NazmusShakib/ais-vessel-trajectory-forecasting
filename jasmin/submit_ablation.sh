#!/bin/bash
# Submit the pre-registered environmental ablation: six arms x N seeds x seven groups.
#
# Each sbatch call submits one 7-task array (one per vessel group) for a single arm and seed, so
# the full grid is ARMS x SEEDS array jobs. Results land in $AIS_RUNS_DIR/<arm>_seed<N>/.
#
# Several seeds per arm is not optional here. The trajectory-only control was measured disagreeing
# with itself by 8.6% CV on Cargo and 4.4% on Port_Service at pilot scale, against an expected
# effect of 2.2% -- so a single run per arm cannot distinguish the effect from the initialisation.
# EnvShip reports five seeds with an ADE standard deviation of 0.2-1.6 m; five is the target here
# too, and the full run should bring the spread into that range.
#
# Usage:  bash jasmin/submit_ablation.sh [model] [mode] [seeds...]
#   bash jasmin/submit_ablation.sh                      # bilstm_attention, full, seeds 1-5
#   bash jasmin/submit_ablation.sh bilstm_attention full 1 2 3
#   DRY_RUN=1 bash jasmin/submit_ablation.sh            # print the sbatch calls, submit nothing
set -euo pipefail
[[ -f run_training.py ]] || { echo 'Run from the training_5_60_v1 folder.' >&2; exit 2; }

model="${1:-bilstm_attention}"; mode="${2:-full}"
shift $(( $# > 2 ? 2 : $# )) || true
seeds=("$@"); [[ ${#seeds[@]} -gt 0 ]] || seeds=(1 2 3 4 5)

# The pre-registered arms. 'none' is the control and must be run on the same seeds as every other
# arm -- comparing an environmental arm against a control trained on one seed is the error this
# whole script exists to prevent. See ENVIRONMENTAL_JOIN_PLAN.md.
arms=(none CUR SSH WAV WND CUR,SSH,WAV,WND)

total=$(( ${#arms[@]} * ${#seeds[@]} ))
echo "$total array jobs (${#arms[@]} arms x ${#seeds[@]} seeds), 7 groups each = $(( total * 7 )) runs"
[[ "${DRY_RUN:-}" ]] && echo '(DRY_RUN: nothing will be submitted)'
for arm in "${arms[@]}"; do
    for seed in "${seeds[@]}"; do
        cmd=(sbatch jasmin/train_gpu.sbatch "$model" "$mode" "$arm" "$seed")
        if [[ "${DRY_RUN:-}" ]]; then printf '  %s\n' "${cmd[*]}"; else "${cmd[@]}"; fi
    done
done
