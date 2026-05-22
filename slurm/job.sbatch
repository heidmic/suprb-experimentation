#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=suprb-moo
#SBATCH --output=stdout/slurm-%j.out

set -e

source /projects/shared/lib/slurm-utils.sh

PROJECT_ROOT=/home/bene/richtig/suprb-experimentation
SLURM_DIR=$PROJECT_ROOT/slurm

cd "$SLURM_DIR"

mkdir -p "$PROJECT_ROOT/results"
mkdir -p "$PROJECT_ROOT/mlruns"

# Dataset persistent auf Node speichern
DATA_DIR=$(stage_data "$PROJECT_ROOT/problems/datasets")

export DATA_DIR
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

srun --export=NONE /run/current-system/sw/bin/nix \
  develop --no-pure-eval --command \
  python -m runs.moo_sc.baseline_tuning.simple_moo_tuning_autologger

# Ergebnisse sichern
save_results "$PROJECT_ROOT/mlruns" "$PROJECT_ROOT/results/mlruns-$SLURM_JOB_ID"