#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=1000

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID"

#Add NS types here
