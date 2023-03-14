#!/usr/bin/env bash

datasets=("concrete_strength" "energy_cool" "combined_cycle_power_plant" "airfoil_self_noise")
scripts=("runs/comparisons/rf.py" "runs/comparisons/decision_tree.py")

for script in ${scripts[@]}; do
    for dataset in ${datasets[@]}; do
    	echo $script $dataset
	export dataset=$dataset
	export experiment=$script
	sbatch slurm/default.sbatch
    done
done



