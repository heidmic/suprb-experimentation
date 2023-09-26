#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute02

USER="krischma"
solution_comp="sas"
dataset="airfoil_self_noise"
experiment="runs/solution_composition/saga_comparison_tuning.py"
JOB_DIR=/data/oc-compute02/$USER/suprb-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -s $solution_comp -j $SLURM_JOB_ID"
