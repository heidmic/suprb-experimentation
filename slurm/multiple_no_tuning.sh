#!/usr/bin/env bash
# filenames=("basic.py" "rf.py" "decision_tree.py")
# for fname in ${filenames[@]}; do 
# 	export filename=$fname
# 	sbatch slurm/no_tuning.sbatch
# done

export filename="basic.py"
echo $filename


export fitness_weight=0.3
export experiment_name="DefaultSingleScaler"
export scaler_type=true
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.1
export experiment_name="FitnessWeightSingleScaler"
export scaler_type=true
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.3
export experiment_name="DefaultDoubleScaler"
export scaler_type=false
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.1
export experiment_name="FitnessWeightDoubleScaler"
export scaler_type=false
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
