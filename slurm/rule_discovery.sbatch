#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
# python $JOB_DIR/$experiment -p $dataset"
python $JOB_DIR/$experiment -p $dataset -t $ns_type -a $archive_pop"
