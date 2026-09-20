#!/bin/bash
#SBATCH --job-name=ais-gpu
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit with sbatch from the training_5_60_v1 folder}"
cd "$SLURM_SUBMIT_DIR"
# Args: model, mode, env blocks (comma-separated or none), seed.
exec bash jasmin/run_group.sh gpu "${1:-bilstm_attention}" "${2:-full}" "${3:-none}" "${4:-}"
