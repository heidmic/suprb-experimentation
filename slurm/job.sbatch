#!/usr/bin/env bash
###############################################################################
# Slurm directives that apply to every array task
###############################################################################
#SBATCH --time=96:00:00                 # 96h wall‑clock limit
#SBATCH --partition=cpu                 # Passende Partition für compute05
#SBATCH --nodelist=oc-compute03         # Ziel-Node auf compute03 geändert
#SBATCH --cpus-per-task=8              # 8 CPUs per array task
#SBATCH --mem=50G                      # 50 GiB RAM per array task
#SBATCH --output=output/%A_%a.out      # stdout file
#SBATCH --error=output/%A_%a.err       # stderr file
#SBATCH --array=0-4                    # ----  five array tasks  ----
#
# Submit: sbatch slurm/job.sbatch
###############################################################################

source /projects/shared/lib/slurm-utils.sh

# Die Trap sichert am Ende alles
trap 'echo "[$(date)] Trap ausgelöst..."; save_results "$SCRATCH" ~"/results/sweep-$SLURM_ARRAY_JOB_ID/task-$SLURM_ARRAY_TASK_ID"' EXIT ERR SIGTERM

PROJECT_DIR="/home/$USER/suprb-experimentation"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
EXPERIMENT="runs/moo_sc/baseline_tuning/simple_moo_tuning_autologger.py"
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

# Zufälliger Versatz beim Start
sleep $(( RANDOM % 21 ))

cd "${PROJECT_DIR}"

# -------------------------------------------------------------------------------
# 1. UMGEBUNG PHASENWEISE VORBEREITEN
# -------------------------------------------------------------------------------
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "[$(date)] Task 0 baut die Devenv-Umgebung exklusiv vor..."
    nix develop ./slurm --no-pure-eval --command python --version
    touch "${PROJECT_DIR}/.devenv_ready"
else
    echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} wartet auf die Umgebung..."
    while [ ! -f "${PROJECT_DIR}/.devenv_ready" ]; do
        sleep 2
    done
    sleep $(( SLURM_ARRAY_TASK_ID * 2 ))
fi

# -------------------------------------------------------------------------------
# 2. NATIVE POSTGRESQL INSTANZ IM SCRATCH INITIALISIEREN & STARTEN
# -------------------------------------------------------------------------------
PG_DATA="$SCRATCH/postgres_data"
PG_SOCKET_DIR="$SCRATCH/postgres"

mkdir -p "$PG_DATA"
mkdir -p "$PG_SOCKET_DIR"

# Trickst Optuna/SQLAlchemy aus, damit es den Socket im richtigen Ordner sucht
export DEVENV_STATE="$SCRATCH"

echo "[$(date)] Initialisiere native PostgreSQL-Datenbank in $PG_DATA..."
nix develop ./slurm --no-pure-eval --command initdb -D "$PG_DATA" --no-locale --auth=trust

echo "[$(date)] Starte PostgreSQL-Server n लिह..."
# Wir starten postgres nativ, lauschen NUR auf dem UNIX-Socket im Scratch (kein TCP-Port-Konflikt!)
nix develop ./slurm --no-pure-eval --command pg_ctl -D "$PG_DATA" -o "-k $PG_SOCKET_DIR -h ''" start

# Dem Server 3 Sekunden Zeit geben, um den Socket zu erstellen
sleep 3

# Erstelle die Standard-Datenbank (falls Optuna sie unter dem System-Usernamen erwartet)
nix develop ./slurm --no-pure-eval --command createdb -h "$PG_SOCKET_DIR" optuna_db

# -------------------------------------------------------------------------------
# 3. EXPERIMENT AUSFÜHREN
# -------------------------------------------------------------------------------
echo "[$(date)] Starte Python-Experiment..."
nix develop ./slurm --no-pure-eval --command env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python "${PROJECT_DIR}/${EXPERIMENT}" \
    -p "${dataset}" \
    -j "${SLURM_ARRAY_JOB_ID}" \
    -o "${OPTIMIZER}"

# -------------------------------------------------------------------------------
# 4. SAUBERES HERUNTERFAHREN
# -------------------------------------------------------------------------------
echo "[$(date)] Stoppe PostgreSQL-Server..."
nix develop ./slurm --no-pure-eval --command pg_ctl -D "$PG_DATA" stop -m fast

# Am Ende räumt Task 0 das Readiness-Flag auf
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    rm -f "${PROJECT_DIR}/.devenv_ready"
fi

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} erfolgreich beendet."