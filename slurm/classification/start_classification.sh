#!/usr/bin/env bash

JOB_DIR=/data/oc-compute03/wehrfabi/my-experimentation
MODELS=("l1" "l2" "elasticnet")
PROBLEMS=("breastcancer" "abalone" "raisin")

for model in "${MODELS[@]}"; do
    for problem in "${PROBLEMS[@]}"; do
        sbatch $JOB_DIR/slurm/classification/class_tuning_${model}_${problem}.sbatch
    done
done
