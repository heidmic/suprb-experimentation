#!/usr/bin/env bash
###############################################################################
# Slurm directives that apply to every array task
###############################################################################
#SBATCH --time=96:00:00                 # 96h wall‑clock limit
#SBATCH --partition=cpu                 # Passende Partition für compute05
#SBATCH --nodelist=oc-compute04         # Ziel-Node auf compute05 geändert
#SBATCH --cpus-per-task=5              # 5 CPUs per array task
#SBATCH --mem=50G                      # 50 GiB RAM per array task
#SBATCH --output=output/%A_%a.out      # stdout file
#SBATCH --error=output/%A_%a.err       # stderr file
#SBATCH --array=0-4                    # ----  five array tasks  ----
#
# Submit: sbatch slurm/job.sbatch
###############################################################################

source /projects/shared/lib/slurm-utils.sh


trap 'echo "[$(date)] Trap ausgelöst..."; save_results "$SCRATCH" ~"/results/sweep-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"' EXIT ERR SIGTERM

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

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "[$(date)] Task 0 baut die Devenv-Umgebung exklusiv vor..."
    # Initialisiert die Umgebung einmal komplett isoliert
    nix develop ./slurm --no-pure-eval --command python --version
    # Erstellt eine temporäre Flagge, dass alles fertig ist
    touch "${PROJECT_DIR}/.devenv_ready"
else
    echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} wartet auf die Umgebung..."
    # Alle anderen Tasks warten, bis Task 0 fertig ist
    while [ ! -f "${PROJECT_DIR}/.devenv_ready" ]; do
        sleep 2
    done
    # Ein kleiner zusätzlicher Versatz, damit nicht alle im selben Sekundenbruchteil zugreifen
    sleep $(( SLURM_ARRAY_TASK_ID * 2 ))
fi

# Jetzt rufen wir den Python-Befehl auf
nix develop ./slurm --no-pure-eval --command env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python "${PROJECT_DIR}/${EXPERIMENT}" \
    -p "${dataset}" \
    -j "${SLURM_ARRAY_JOB_ID}" \
    -o "${OPTIMIZER}"

# Am Ende räumt Task 0 die Flagge wieder auf
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    rm -f "${PROJECT_DIR}/.devenv_ready"
fi