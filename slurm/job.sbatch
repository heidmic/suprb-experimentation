#!/usr/bin/env bash
###############################################################################
# Slurm directives that apply to every array task
###############################################################################
#SBATCH --time=96:00:00                 # 96h wall‑clock limit (Vorsicht: Default ist 1h, stelle sicher, dass 96h erlaubt sind!)
#SBATCH --partition=bigmem              # Passende Partition für oc-compute03 (Alternativ: cpu)
#SBATCH --nodelist=oc-compute03        # Bleibe auf diesem spezifischen Node
#SBATCH --cpus-per-task=5              # 5 CPUs per array task
#SBATCH --mem=20G                      # 20GiB RAM per array task
#SBATCH --output=output/%A_%a.out      # stdout file  (%A=array‑id, %a=task‑id)
#SBATCH --error=output/%A_%a.err       # stderr file
#SBATCH --array=0-4                    # ----  five array tasks  ----
#
# Submit: sbatch moo_experiment.sbatch
###############################################################################

# Hilfsfunktionen des Clusters laden (wichtig für save_results)
source /projects/shared/lib/slurm-utils.sh

# 1. Projekt-Verzeichnis im Home-Ordner definieren
# Ersetze <dein_rz_id> mit deinem tatsächlichen Benutzernamen!
PROJECT_DIR="/home/$USER/suprb-experimentation"

export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
EXPERIMENT="runs/moo_sc/baseline_tuning/simple_moo_tuning.py"
OPTIMIZER="nsga2"

# 2. Map array‑task ID → dataset name
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

# 3. Wechsel in das Projektverzeichnis
cd "${PROJECT_DIR}"

# 4. Experiment über die neue Nix-Flake-Umgebung ausführen
# Wichtig: Dein Python-Skript sollte Ausgaben in den Ordner schreiben, der in $SCRATCH hinterlegt ist
nix develop ./slurm --no-pure-eval --command python "${PROJECT_DIR}/${EXPERIMENT}" \
    -p "${dataset}" \
    -j "${SLURM_ARRAY_JOB_ID}" \
    -o "${OPTIMIZER}" \
    --output "$SCRATCH"

# 5. Ergebnisse sichern
# Da $SCRATCH nach dem Job gelöscht wird, kopiert dieser Befehl alles zurück in dein Home-Verzeichnis
save_results "$SCRATCH" "~/results/sweep-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"