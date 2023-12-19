#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute03

USER="krischma"
script="logging_output_scripts/stat_analysis.py"
JOB_DIR=/data/oc-compute02/$USER/suprb-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$script calvo --latex --check-mcmc"