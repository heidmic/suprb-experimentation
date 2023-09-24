#!/usr/bin/env bash
#SBATCH --time=73:00:00
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=1000
USER=niemeith
JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
#change this line to the python code you want to execute
EXPERIMENT=runs/hyperellipsoidal-run/main.py
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH python $JOB_DIR/$EXPERIMENT"


