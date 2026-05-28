#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G

JOB_DIR=/home/$USER/suprb-experimentation
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
experiment="runs/comparisons/suprb_tuning.py"
dataset="airfoil_self_noise"
study_name="SupRB-Airfoil"

nix develop --no-pure-eval --command python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID 
