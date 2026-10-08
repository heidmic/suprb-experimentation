#!/usr/bin/env bash
#SBATCH --time=96:00:00                 # 96h wall‑clock limit
#SBATCH --partition=cpu                # queue / partition

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
export PYTHONPATH="${PYTHONPATH}:${JOB_DIR}"
FILE_TO_RUN="logging_output_scripts/one_time_run.py"

srun nix-shell "${JOB_DIR}/slurm/default.nix" --command "
  python ${JOB_DIR}/${FILE_TO_RUN} \
"