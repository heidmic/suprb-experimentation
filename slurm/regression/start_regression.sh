#!/usr/bin/env bash

JOB_DIR=/data/oc-compute03/wehrfabi/my-experimentation
MODELS=("lasso" "ridge" "elasticNet")
PROBLEMS=("concrete_strength" "airfoil_self_noise" "combined_cycle_power_plant" "energy_heat")

for model in "${MODELS[@]}"; do
    for problem in "${PROBLEMS[@]}"; do
        sbatch $JOB_DIR/slurm/regression/regress_tuning_${model}_${problem}.sbatch
    done
done