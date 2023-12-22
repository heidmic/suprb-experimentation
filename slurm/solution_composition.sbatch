#!/usr/bin/env bash

#SBATCH --job-name=solution_composition_${$dataset}_${optimizer}
#SBATCH --partition=epyc
#SBATCH --time=96:00:00
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

JOB_DIR=/hpc/gpfs2/home/u/sraroman/suprb-experimentation 
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -o $optimizer"
