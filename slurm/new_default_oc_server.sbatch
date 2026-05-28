#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/%A_%a.out      # stdout file
#SBATCH --error=output/%A_%a.err
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute04

PROJECT_DIR="/home/$USER/suprb-experimentation"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
EXPERIMENT="runs/comparisons/suprb_tuning.py"
dataset="airfoil_self_noise"
study_name="SupRB-Airfoil"

cd "${PROJECT_DIR}"

nix develop ./slurm --no-pure-eval --command env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python "${PROJECT_DIR}/${EXPERIMENT}" \
    --problem "$dataset" \
    --job_id "$SLURM_JOB_ID" 
