#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=gpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID"
