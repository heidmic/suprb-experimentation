#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/jsaon_ridge-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G

JOB_DIR=/data/oc-compute03/wehrfabi/my-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/runs/local_models/json_tuning.py -p concrete_strength -l ridge"
