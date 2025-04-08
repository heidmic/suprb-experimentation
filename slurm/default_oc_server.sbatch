#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute03

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
experiment="runs/comparisons/suprb_all_tuning.py"
dataset="airfoil_self_noise"
study_name="SupRB-Airfoil"

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID -n $study_name"
