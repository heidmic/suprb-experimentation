#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000

USER = # TODO
JOB_DIR=/data/oc-compute01/$USER/suprb-experimentation
EXPERIMENT=runs/run.py
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH python $JOB_DIR/$EXPERIMENT"
