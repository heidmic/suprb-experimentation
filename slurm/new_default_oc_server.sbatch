#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute04

PROJECT_DIR="/home/$USER/suprb-experimentation"
export PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
experiment="runs/comparisons/suprb_tuning.py"
dataset="airfoil_self_noise"
study_name="SupRB-Airfoil"

cd "${PROJECT_DIR}"

nix develop ./slurm --no-pure-eval --command env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python "${PROJECT_DIR}/${EXPERIMENT}" \
python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID -n $study_name"
python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID -r $rules_amount -f"
        -p "${dataset}" \
