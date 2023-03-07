#!/usr/bin/env bash

datasets=("concrete_strength" "energy_heat" "energy_cool" "parkinson_total" "parkinson_motor" "gas_turbine" "combined_cycle_power_plant" "concrete_strength" "airfoil_self_noise")
scripts=("runs/comparisons/rf.py" "runs/comparisons/decision_tree.py" "runs/comparisons/suprb_tuning.py")

for script in ${scripts[@]}; do
    for dataset in ${datasets[@]}; do
    	echo $script $dataset
	export dataset=$dataset
	export experiment=$script
	sbatch slurm/default.sbatch
    done
done



