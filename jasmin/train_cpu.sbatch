#!/bin/bash
#SBATCH --job-name=ais-cpu
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err
#SBATCH --no-requeue
# Supply --account=YOUR_VALID_ACCOUNT when calling sbatch (see useraccounts).
set -euo pipefail
: "${SLURM_SUBMIT_DIR:?Submit with sbatch from the training_5_60_v1 folder}"
cd "$SLURM_SUBMIT_DIR"
exec bash jasmin/run_group.sh cpu "${1:-baselines}" "${2:-full}"
