#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute03/volgerbe/output/output-%j.txt
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=1000

USER=volgerbe
JOB_DIR=/data/oc-compute03/$USER/suprb2-experimentation
EXPERIMENT=runs/examples/ns_tuning.py
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH python $JOB_DIR/$EXPERIMENT"
