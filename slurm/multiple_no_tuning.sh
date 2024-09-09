#!/usr/bin/env bash
filenames=("basic.py" "rf.py" "decision_tree.py")
for fname in ${filenames[@]}; do 
	export filename=$fname
	sbatch slurm/no_tuning.sbatch
done
