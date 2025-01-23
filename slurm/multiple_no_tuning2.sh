#!/usr/bin/env bash
# filenames=("basic.py" "rf.py" "decision_tree.py")
# for fname in ${filenames[@]}; do 
# 	export filename=$fname
# 	sbatch slurm/no_tuning.sbatch
# done

JOB_DIR=/home/giemo/suprb-experimentation

export filename="basic.py"
echo $filename


for i in {1..64}
do

export fitness_weight=0.3
export experiment_name="DefaultSingleScaler10"
export scaler_type=true
export random_state=$i
echo $fitness_weight
echo $experiment_name
echo $scaler_type
echo $random_state
sbatch slurm/no_tuning.sbatch
sleep 2

export fitness_weight=0.1
export experiment_name="FitnessWeightSingleScaler10"
export scaler_type=true
export random_state=$i
echo $fitness_weight
echo $experiment_name
echo $scaler_type
echo $random_state
sbatch slurm/no_tuning.sbatch
sleep 2

export fitness_weight=0.3
export experiment_name="DefaultDoubleScaler10"
export scaler_type=false
export random_state=$i
echo $fitness_weight
echo $experiment_name
echo $scaler_type
echo $random_state
sbatch slurm/no_tuning.sbatch
sleep 2

export fitness_weight=0.1
export experiment_name="FitnessWeightDoubleScaler10"
export scaler_type=false
export random_state=$i
echo $fitness_weight
echo $experiment_name
echo $scaler_type
echo $random_state
sbatch slurm/no_tuning.sbatch
sleep 2

done
