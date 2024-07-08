#!/usr/bin/env bash
datasets=("concrete_strength" "energy_cool" "combined_cycle_power_plant" "airfoil_self_noise" "parkinson_total" "protein_structure")

for dset in ${datasets[@]}; do 
	export dataset=$dset
	export experiment="runs/comparisons/suprb_all_tuning.py"
	export study_name=$dset
	sbatch slurm/all_tuning.sbatch
	sleep 30
done
