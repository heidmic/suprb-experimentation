#!/usr/bin/env bash
###############################################################################
# Slurm directives that apply to every array task
###############################################################################
#SBATCH --time=96:00:00                 # 96h wall‑clock limit
#SBATCH --partition=cpu                 # Passende Partition für compute05
#SBATCH --nodelist=oc-compute05         # Ziel-Node auf compute05 geändert
#SBATCH --cpus-per-task=5              # 5 CPUs per array task
#SBATCH --mem=20G                      # 20GiB RAM per array task
#SBATCH --output=output/%A_%a.out      # stdout file
#SBATCH --error=output/%A_%a.err       # stderr file
#SBATCH --array=0-4                    # ----  five array tasks  ----
#
# Submit: sbatch slurm/moo_experiment.sbatch
###############################################################################

source /projects/shared/lib/slurm-utils.sh

PROJECT_DIR="/home/$USER/suprb-experimentation"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
EXPERIMENT="runs/moo_sc/baseline_tuning/simple_moo_tuning.py"
OPTIMIZER="nsga2"

datasets=(
  airfoil_self_noise
  concrete_strength
  combined_cycle_power_plant
  protein_structure
  parkinson_total
)

dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
study_name="${OPTIMIZER} Tuning - ${dataset}"

echo "[$(date)] Tuning optimizer '${OPTIMIZER}' on dataset '${dataset}' (array task ${SLURM_ARRAY_TASK_ID})"

sleep $(( RANDOM % 21 ))

cd "${PROJECT_DIR}"

# Führt das Flake im slurm-Unterordner aus
nix develop ./slurm --no-pure-eval --command env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python "${PROJECT_DIR}/${EXPERIMENT}" \
    -p "${dataset}" \
    -j "${SLURM_ARRAY_JOB_ID}" \
    -o "${OPTIMIZER}" \
    --output "$SCRATCH"

save_results "$SCRATCH" "~/results/sweep-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"