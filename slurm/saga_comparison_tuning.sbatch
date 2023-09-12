#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
SBATCH --output=output/output-%j.txt
SBATCH --cpus-per-task=5
SBATCH --mem=20G

solution_comp = "saga1"
experiment="runs/comparisons/saga_comparison.py"
JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -s $solution_comp -j $SLURM_JOB_ID"
