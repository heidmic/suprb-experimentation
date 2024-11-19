#!/usr/bin/env bash

#SBATCH --job-name="suprb"
#SBATCH --partition=epyc
#SBATCH --time=120:00:00
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=8 
#SBATCH --mem-per-cpu=5G

module purge
module load anaconda
conda activate venv

JOB_DIR=/hpc/gpfs2/home/u/sraroman/suprb-experimentation 
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
srun python $JOB_DIR/$experiment -p $dataset -o $optimizer
