#!/usr/bin/env bash

#SBATCH --job-name="rule_discovery_"$dataset"_"$ns_type"_"$archive_pop
#SBATCH --partition=epyc
#SBATCH --time=120:00:00
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G

conda_exec=$(which conda)
conda_env="/hpc/gpfs2/scratch/u/sraroman/.conda/envs/venv"
source $conda_exec activate $conda_env

JOB_DIR=/hpc/gpfs2/home/u/sraroman/suprb-experimentation 
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
srun python $JOB_DIR/$experiment -p $dataset -t $ns_type -a $archive_pop

source $conda_exec deactivate

