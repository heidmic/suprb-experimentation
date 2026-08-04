#!/usr/bin/env bash

# called by launch_grid.sh, which sets DATASET, N_ITER, N_RULES, N_INITIAL_RULES via --export
# FLAG selects which script/mode to run:
#   0 -> suprb_default --no-tune
#   1 -> suprb_default --tune
#   2 -> suprb_moo
#   3 -> suprb_ns --use_current_population true
#   4 -> suprb_ns --use_current_population false


#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=96:00:00             #format: HH:MM:SS, tuning 24h + evaluation 48h + mlflow 24h = 96h, eval für 96,128_16_*: 48h pro seed
#SBATCH --job-name=suprb_default   # Job-Name, %x in --output
#SBATCH --output=stdout/%x-%j.out                      # %x = job-name, %j = job-id

source /projects/shared/lib/slurm-utils.sh

JOB_DIR=/home/huettiju/data/oc-compute03/suprb-experimentation
FLAG="${FLAG:-0}" #default 0 if not set

case "${FLAG}" in
  0) 
    echo "Running suprb_default --no-tune" 
    experiment="runs/mybenchmark/suprb_default" #path 
    EXTRA_ARGS="--no-tune"
    ;;
  1) 
    echo "Running suprb_default --tune" 
    experiment="runs/mybenchmark/suprb_default" #path 
    EXTRA_ARGS="--tune"
    ;;
  2) 
    echo "Running suprb_moo" 
    experiment="runs/mybenchmark/suprb_moo" #path 
    EXTRA_ARGS=""
    ;;
  3) 
    echo "Running suprb_ns --use_current_population true" 
    experiment="runs/mybenchmark/suprb_ns" #path 
    EXTRA_ARGS="-a true" #use_current_population true
    ;;
  4) 
    echo "Running suprb_ns --use_current_population false" 
    experiment="runs/mybenchmark/suprb_ns" #path 
    EXTRA_ARGS="-a false" #use_current_population false
    ;;
  *) echo "Unknown FLAG: ${FLAG}" ; exit 1 ;;
esac

echo "FLAG=${FLAG} -> experiment=${experiment} args='${EXTRA_ARGS}'"

cd ${JOB_DIR} #where the nix flake is

nix develop --no-pure-eval --command \
  env PYTHONPATH=${JOB_DIR}/runs/mybenchmark:${JOB_DIR} \
  uv run python ${JOB_DIR}/${experiment} \
    -p ${DATASET} \
    -j ${SLURM_JOB_ID} \
    -n ${N_ITER} \
    -r ${N_RULES} \
    -i ${N_INITIAL_RULES} \
    ${EXTRA_ARGS}


#save_results results  

