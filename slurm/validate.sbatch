#!/usr/bin/env bash
#SBATCH --job-name=labelling-validate
#SBATCH --partition=gammaweb
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/logs/validate-%j.out
#SBATCH --error=slurm/logs/validate-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the labelling repository}"
export PYTHONUNBUFFERED=1
./.venv/bin/labelling validate
