#!/usr/bin/env bash

#SBATCH --job-name=rule_discovery_${$dataset}_${ns_type}_${archive_pop}
#SBATCH --partition=epyc
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

JOB_DIR=/hpc/gpfs2/home/u/sraroman/suprb-experimentation 
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -t $ns_type -a $archive_pop"
