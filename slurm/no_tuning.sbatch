#!/usr/bin/env bash

#SBATCH --job-name="suprb"
#SBATCH --partition=epyc
#SBATCH --time=120:00:00
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5G

module purge
module load anaconda
conda activate venv

export OMP_NUM_THREADS=1

JOB_DIR=/hpc/gpfs2/home/u/sraroman/suprb-experimentation
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
srun python $JOB_DIR/runs/run_without_tuning/$filename -n $exp_name -w $weight -s $scaler_type
