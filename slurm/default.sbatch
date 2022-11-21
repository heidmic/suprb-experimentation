#!/usr/bin/env bash
#SBATCH --time=35:00:00
#SBATCH --partition=cpu-prio
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=1G

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
EXPERIMENT=runs/rule_discovery/evolution_strategy.py
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH python $JOB_DIR/$EXPERIMENT -p $dataset -j $SLURM_JOB_ID"
