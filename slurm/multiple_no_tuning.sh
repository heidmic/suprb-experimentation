#!/usr/bin/env bash
filenames = ("basic" "rf", decision_tree)
for fname in ${filenames[@]}; do 
	export filename=$fname
		sbatch slurm/basic.sbatch
	done
done
