#!/usr/bin/env bash
#SBATCH --time=73:00:00
#SBATCH --partition=cpu
#SBATCH --output=/data/oc-compute03/niemeith/suprb-experimentation/output/evaluation-ghe-CS-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute03

USER=niemeith
JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
EXPERIMENT=runs/hyperellipsoid_tests/evaluation/evaluation_single/ghe/CS_ghe_evaluation.py

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH python $JOB_DIR/$EXPERIMENT"