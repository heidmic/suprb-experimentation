#!/usr/bin/env bash

datasets=("concrete_strength" "energy_cool" "combined_cycle_power_plant" "airfoil_self_noise" "german.numer" "fourclass_scale" "meta" "chscase_foot")

scripts=("runs/solution_composition/solution_composition_tuning_updated.py")

optimizers=('GeneticAlgorithm' 'ArtificialBeeColonyAlgorithm' 'AntColonyOptimization' 'GreyWolfOptimizer' 'ParticleSwarmOptimization' "RandomSearch")

for script in ${scripts[@]}; do
    for dataset in ${datasets[@]}; do
		export dataset=$dataset
		export experiment=$script
		for opt in ${optimizers[@]}; do 
			export optimizer=$opt
	
			echo $script $dataset $opt

			# sbatch slurm/solution_composition.sbatch
		done
    done
done
