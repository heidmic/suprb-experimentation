#!/usr/bin/env bash

#SBATCH --job-name="suprb"
#SBATCH --partition=epyc
#SBATCH --time=144:00:00
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=5G

module purge
module load anaconda
conda activate venv

export OMP_NUM_THREADS=1

JOB_DIR=/hpc/gpfs2/home/u/sraroman/suprb-experimentation
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
srun python $JOB_DIR/runs/run_without_tuning/$filename -n $experiment_name -w $fitness_weight -s $scaler_type -i $random_state
