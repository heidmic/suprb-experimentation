#!/usr/bin/env bash
datasets=("concrete_strength" "energy_cool" "combined_cycle_power_plant" "airfoil_self_noise" "parkinson_total" "protein_structure")

for dset in ${datasets[@]}; do 
	export dataset=$dset

	############################ SOLUTION COMPOSITON ################################

	export experiment="runs/solution_composition/solution_composition_tuning_updated.py"
	optimizers=('GeneticAlgorithm' 'ArtificialBeeColonyAlgorithm' 'AntColonyOptimization' 'GreyWolfOptimizer' 'ParticleSwarmOptimization' "RandomSearch")

	for opt in ${optimizers[@]}; do 
		export optimizer=$opt

		echo $experiment $dataset $opt
		sbatch slurm/solution_composition.sbatch
		sleep 10
	done

	# ############################ RULE DISCOVERY - ES ################################

	# export experiment="runs/rule_discovery/evolution_strategy.py"
	# echo $experiment $dataset
	# sbatch slurm/rule_discovery.sbatch

	# ############################ RULE DISCOVERY - RS ################################

	# export experiment="runs/rule_discovery/random_search.py"
	# echo $experiment $dataset
	# sbatch slurm/rule_discovery.sbatch

	# ############################ RULE DISCOVERY - NS ################################

	# export experiment="runs/rule_discovery/novelty_search.py"
	# ns_types=("NS" "MCNS" "NSLC")

	# for ns in ${ns_types[@]}; do
	# 	export ns_type=$ns
	# 	export archive_pop=true
	# 	echo $experiment $dataset $ns_type $archive_pop
	# 	sbatch slurm/rule_discovery_ns.sbatch

	# 	export archive_pop=false
	# 	echo $experiment $dataset $ns_type $archive_pop
	# 	sbatch slurm/rule_discovery_ns.sbatch
	# done
done
