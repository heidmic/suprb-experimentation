#!/usr/bin/env bash

datasets=("concrete_strength") #"energy_cool" "combined_cycle_power_plant" "airfoil_self_noise" "LEV", "german.numer" "laser" "fourclass_scale" "machine_cpu" "boston" "meta" "chscase_foot")

scripts=("runs/rule_discovery/evolution_strategy.py runs/rule_discovery/random_search.py")
# scripts=("runs/rule_discovery/novelty_search.py")

ns_types=("NS" "MCNS" "NSLC")

for script in ${scripts[@]}; do
    for dataset in ${datasets[@]}; do
		export dataset=$dataset
		export experiment=$script

		# for ns in ${ns_types[@]}; do
		# 	export ns_type=$ns
		# 	export archive_pop=true
		# 	sbatch slurm/rule_discovery.sbatch

		# 	sleep 5
		# 	export archive_pop=false
		# 	sbatch slurm/rule_discovery.sbatch
		#	echo $script $dataset $ns_type $archive_pop
		# done

		echo $script $dataset $ns_type

	sbatch slurm/rule_discovery.sbatch
    done
done



