#!/usr/bin/env bash

export dataset="chscase_census6"

############################ SOLUTION COMPOSITON ################################

export experiment="runs/solution_composition/solution_composition_tuning_updated.py"
optimizers=('GeneticAlgorithm' 'ArtificialBeeColonyAlgorithm' 'AntColonyOptimization' 'GreyWolfOptimizer' 'ParticleSwarmOptimization' "RandomSearch")

for opt in ${optimizers[@]}; do 
	export optimizer=$opt

	echo $experiment $dataset $opt
	sbatch slurm/solution_composition.sbatch
done

############################ RULE DISCOVERY - ES ################################

export experiment="runs/rule_discovery/evolution_strategy.py"
echo $experiment $dataset
sbatch slurm/rule_discovery.sbatch

############################ RULE DISCOVERY - RS ################################

export experiment="runs/rule_discovery/random_search.py"
echo $experiment $dataset
sbatch slurm/rule_discovery.sbatch

############################ RULE DISCOVERY - NS ################################

export experiment="runs/rule_discovery/novelty_search.py"
ns_types=("NS" "MCNS" "NSLC")

for ns in ${ns_types[@]}; do
	export ns_type=$ns
	export archive_pop=true
	echo $experiment $dataset $ns_type $archive_pop
	sbatch slurm/rule_discovery.sbatch

	export archive_pop=false
	echo $experiment $dataset $ns_type $archive_pop
	sbatch slurm/rule_discovery.sbatch
done
