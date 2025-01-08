#!/usr/bin/env bash
# filenames=("basic.py" "rf.py" "decision_tree.py")
# for fname in ${filenames[@]}; do 
# 	export filename=$fname
# 	sbatch slurm/no_tuning.sbatch
# done

export filename="basic.py"
echo $filename


export fitness_weight=0.3
export experiment_name="DefaultSingleScaler10"
export scaler_type=true
export n_iter=10
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.1
export experiment_name="FitnessWeightSingleScaler10"
export scaler_type=true
export n_iter=10
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.3
export experiment_name="DefaultDoubleScaler10"
export scaler_type=false
export n_iter=10
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.1
export experiment_name="FitnessWeightDoubleScaler10"
export scaler_type=false
export n_iter=10
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch



export fitness_weight=0.3
export experiment_name="DefaultSingleScaler64"
export scaler_type=true
export n_iter=64
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.1
export experiment_name="FitnessWeightSingleScaler64"
export scaler_type=true
export n_iter=64
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.3
export experiment_name="DefaultDoubleScaler64"
export scaler_type=false
export n_iter=64
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
sleep 10

export fitness_weight=0.1
export experiment_name="FitnessWeightDoubleScaler64"
export scaler_type=false
export n_iter=64
echo $fitness_weight
echo $experiment_name
echo $scaler_type
sbatch slurm/no_tuning.sbatch
