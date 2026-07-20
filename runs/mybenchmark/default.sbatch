#!/usr/bin/env bash

# called by launch_grid.sh, which sets DATASET, N_ITER, N_RULES, N_INITIAL_RULES via --export

#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00             #format: HH:MM:SS, tuning 24h + evaluation 48h + mlflow 24h = 96h, eval für 96,128_16_*: 48h pro seed
#SBATCH --job-name=suprb_default   # Job-Name, %x in --output
#SBATCH --output=stdout/%x-%j.out                      # %x = job-name, %j = job-id

source /projects/shared/lib/slurm-utils.sh

JOB_DIR=/home/huettiju/data/oc-compute03/suprb-experimentation
#export PYTHONPATH=${PYTHONPATH}:${JOB_DIR} 
#export PYTHONPATH=${JOB_DIR}/runs/mybenchmark:${PYTHONPATH}:${JOB_DIR} # to find lokal evaluation file

experiment="runs/mybenchmark/suprb_default" #path 

TUNE_ARG="--no-tune"
[[ "${FLAG:-0}" == "1" ]] && TUNE_ARG="--tune"

cd ${JOB_DIR} #where the nix flake is

nix develop --no-pure-eval --command \
  env PYTHONPATH=${JOB_DIR}/runs/mybenchmark:${JOB_DIR} \
  uv run python ${JOB_DIR}/${experiment} \
    -p ${DATASET} \
    -j ${SLURM_JOB_ID} \
    -n ${N_ITER} \
    -r ${N_RULES} \
    -i ${N_INITIAL_RULES} \
    ${TUNE_ARG}


#save_results results  

