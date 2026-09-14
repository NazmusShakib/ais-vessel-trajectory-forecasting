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
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
echo "Job $SLURM_JOB_ID | $model | $group | $mode | $device"
args=(run_training.py "$model" --mode "$mode" --groups "$group" --device "$device"
      --data-dir "$AIS_DATASET_DIR" --runs-dir "$AIS_RUNS_DIR" --jobs-dir "$AIS_JOBS_DIR")
# Check metadata and the actual GPU allocation before spending time training.
"$AIS_PYTHON" -u "${args[@]}" --check-config
exec "$AIS_PYTHON" -u "${args[@]}"
