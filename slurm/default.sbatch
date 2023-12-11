#!/usr/bin/env bash
#SBATCH --time=48:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G

JOB_DIR=/data/oc222/$USER/suprb-experimentation
srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID -r $rules_amount -f $filter_subpopulation -e $experience_calculation"
