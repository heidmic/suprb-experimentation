#!/usr/bin/env bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=mlflow_ui
#SBATCH --output=stdout/mlflow_ui-%j.out

source /projects/shared/lib/slurm-utils.sh
tunnel_info 5000

JOB_DIR=/home/huettiju/data/oc-compute03/suprb-experimentation
cd "$JOB_DIR" 

srun nix develop --no-pure-eval --command bash -c \
  'uv run mlflow ui --backend-store-uri ./runs/mybenchmark/mlruns --host 0.0.0.0 --port 5000 --workers 1 --gunicorn-opts "--timeout 120"'

