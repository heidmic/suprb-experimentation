#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/models_l1-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G

JOB_DIR=/data/oc-compute03/wehrfabi/my-experimentation
MODELS=("l1")
PROBLEMS=("breastcancer" "abalone" "raisin")

for model in "${MODELS[@]}"; do
    for problem in "${PROBLEMS[@]}"; do
        srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
        python $JOB_DIR/runs/local_models/class_tuning.py -p $problem -l $model"
    done
done