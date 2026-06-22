#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=stdout/dashboard-%j.out

source /projects/shared/lib/slurm-utils.sh
tunnel_info 8888

JOB_DIR=/home/huettiju/data/oc-compute03/suprb-experimentation
DB_PATH="${JOB_DIR}/runs/mybenchmark/optuna_dbs/airfoil_self_noise__ni32__nr8__nir0.db"

cd "$JOB_DIR" 

srun nix develop --no-pure-eval --command bash -c \
  "uv run optuna-dashboard sqlite:///${DB_PATH} --host 0.0.0.0 --port 8888" 

