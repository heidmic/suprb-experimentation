#!/usr/bin/env bash
#SBATCH --time=73:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1000


JOB_DIR=/data/oc-compute03/hubanton/4_MPR/suprb-experimentation
EXPERIMENT=runs/solution_composition/rep_tuning.py
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH python $JOB_DIR/$EXPERIMENT"


