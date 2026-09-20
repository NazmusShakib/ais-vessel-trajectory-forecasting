#!/bin/bash
# Run within a Slurm allocation; never starts training on a login/sci server.
set -euo pipefail
: "${SLURM_JOB_ID:?This wrapper must run within a Slurm job}"
if [[ ! -f run_training.py || ! -f jasmin/config.sh ]]; then
    echo 'Submit from training_5_60_v1 after copying/editing jasmin/config.sh.example to jasmin/config.sh.' >&2
    exit 2
fi
source jasmin/config.sh
for name in AIS_PYTHON AIS_DATASET_DIR AIS_RUNS_DIR AIS_JOBS_DIR; do
    value="${!name:-}"
    if [[ -z "$value" || "$value" == *REPLACE_WITH* ]]; then
        echo "Set $name in jasmin/config.sh first." >&2
        exit 2
    fi
done
[[ -x "$AIS_PYTHON" ]] || { echo "Python executable missing: $AIS_PYTHON" >&2; exit 2; }
for file in manifest.json shard_index.csv; do
    [[ -f "$AIS_DATASET_DIR/$file" ]] || { echo "Missing dataset file: $AIS_DATASET_DIR/$file" >&2; exit 2; }
done
[[ -d "$AIS_DATASET_DIR/groups" ]] || { echo 'Dataset groups/ directory missing.' >&2; exit 2; }
device="${1:?cpu or gpu}"; model="${2:?model name}"; mode="${3:?smoke, pilot or full}"
# Environmental blocks arrive COMMA-separated -- 'CUR,SSH,WAV,WND', or 'none' for a
# trajectory-only run. Commas rather than spaces because this value crosses sbatch, an exec and
# an array assignment: a space-separated string arrives as one argument and argparse then rejects
# 'CUR SSH WAV WND' as an invalid choice. That exact bug silently lost an ablation arm locally.
blocks="${4:-none}"; seed="${5:-}"
case "$device" in cpu|gpu) ;; *) echo 'Device must be cpu or gpu.' >&2; exit 2;; esac
case "$model" in baselines|bilstm_attention|bilstm_only|transformer) ;; *) echo 'Unknown model.' >&2; exit 2;; esac
case "$mode" in smoke|pilot|full) ;; *) echo 'Unknown mode.' >&2; exit 2;; esac
if [[ "$device" == gpu && "$model" == baselines ]]; then
    echo 'Use train_cpu.sbatch for baselines.' >&2; exit 2
fi
vessel_groups=(Cargo Tanker Passenger Port_Service Research_Offshore Fishing Unknown)
index="${SLURM_ARRAY_TASK_ID:-0}"
[[ "$index" =~ ^[0-6]$ ]] || { echo 'Array index must be 0 through 6.' >&2; exit 2; }
group="${vessel_groups[$index]}"
env_args=()
if [[ -n "$blocks" && "$blocks" != none ]]; then
    IFS=',' read -r -a block_list <<< "$blocks"
    for block in "${block_list[@]}"; do
        case "$block" in CUR|SSH|WAV|WND) ;; *) echo "Unknown env block: $block" >&2; exit 2;; esac
    done
    : "${AIS_ENV_SIDECAR:?Set AIS_ENV_SIDECAR in jasmin/config.sh to run environmental blocks}"
    [[ "$AIS_ENV_SIDECAR" != *REPLACE_WITH* ]] || { echo 'Set AIS_ENV_SIDECAR in jasmin/config.sh first.' >&2; exit 2; }
    [[ -d "$AIS_ENV_SIDECAR" ]] || { echo "Sidecar directory missing: $AIS_ENV_SIDECAR" >&2; exit 2; }
    env_args=(--env-blocks "${block_list[@]}" --env-sidecar "$AIS_ENV_SIDECAR")
fi
seed_args=()
[[ -n "$seed" ]] && seed_args=(--seed "$seed")
# Results are separated by arm and seed. Every ablation arm needs several seeds -- the control's
# own seed-to-seed spread was measured at 8.6% CV for Cargo, larger than the effect being tested --
# so runs must not land in one directory where they cannot be told apart.
label="${blocks//,/+}"
[[ -n "$seed" ]] && label="${label}_seed${seed}"
runs_dir="$AIS_RUNS_DIR/$label"

export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
echo "Job $SLURM_JOB_ID | $model | $group | $mode | $device | blocks=$blocks | seed=${seed:-default}"
args=(run_training.py "$model" --mode "$mode" --groups "$group" --device "$device"
      --data-dir "$AIS_DATASET_DIR" --runs-dir "$runs_dir" --jobs-dir "$AIS_JOBS_DIR")
# Appended only when non-empty. Expanding an empty array as "${arr[@]}" is an unbound-variable
# error under `set -u` on older bash, and the array is empty for exactly one case: the
# trajectory-only control, which is the arm every other arm is measured against. Caught by a dry
# run before submission; it would otherwise have failed only on JASMIN, only on the control.
if [[ ${#env_args[@]} -gt 0 ]]; then args+=("${env_args[@]}"); fi
if [[ ${#seed_args[@]} -gt 0 ]]; then args+=("${seed_args[@]}"); fi
# Check metadata and the actual GPU allocation before spending time training.
"$AIS_PYTHON" -u "${args[@]}" --check-config
exec "$AIS_PYTHON" -u "${args[@]}"
