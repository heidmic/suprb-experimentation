#!/usr/bin/env bash

script="runs/comparisons/suprb_tuning.py"

datasets=("concrete_strength" "energy_cool" "combined_cycle_power_plant" "airfoil_self_noise")
filter_subpop=("FilterSubpopulation" "NBestFitness" "NRandom" "RouletteWheel")
experience_calc=("ExperienceCalculation" "CapExperience" "CapExperienceWithDimensionality")

rules_amount_start=1
rules_amount_end=5

export experiment=$script

# Loop through datasets
for dataset in ${datasets[@]}; do
	export dataset=$dataset

	# Loop through filter_subpopulation
	for filter in  ${filter_subpop[@]}; do
		export filter_subpopulation=$filter

		# Loop through experience_calculation
		for experience in  ${experience_calc[@]}; do
			export experience_calculation=$experience

			# Loop through rules_amount
			for ((k=rules_amount_start;k<=rules_amount_end;k++)); do
				export rules_amount=$k

				sbatch slurm/default.sbatch
				echo "$JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID -r $rules_amount -f $filter_subpopulation -e $experience_calculation"
				sleep 5
			done
		done
	done
done
