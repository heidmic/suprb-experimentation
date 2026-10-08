#!/usr/bin/env bash
###############################################################################
# worker.sbatch
#
# One worker task. Very lightweight — just runs the Python experiment and
# writes its Optuna trial results to the shared PostgreSQL study.
#
# NOT submitted directly. The coordinator (start_study.sbatch) submits this
# as an array job and passes all required variables via --export.
#
# Environment variables expected from coordinator:
#   OPTIMIZER        e.g. "nsga2"
#   DATASET          e.g. "airfoil_self_noise"
#   PG_SOCKET_DIR    path to the shared PostgreSQL UNIX socket directory
#   STUDY_NAME       the Optuna study name to attach to
#   TIMEOUT_HOURS    per-worker time budget (hours, float ok)
###############################################################################
#SBATCH --job-name=suprb_worker
#SBATCH --partition=cpu
#SBATCH --nodelist=oc-compute04
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=80:00:00
#SBATCH --output=output/worker_%A_%a.out
#SBATCH --error=output/worker_%A_%a.err
# NOTE: --array and --export are set by the coordinator at submission time.

source /projects/shared/lib/slurm-utils.sh

PROJECT_DIR="/home/$USER/suprb-experimentation"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
EXPERIMENT="runs/moo_sc/baseline_tuning/simple_moo_tuning_database.py"

# Save results to per-worker scratch, then copy to shared results on exit
trap '
    echo "[$(date)] Worker ${SLURM_ARRAY_TASK_ID} trap: saving results..."
    save_results "$SCRATCH" ~"/results/sweep-${SLURM_ARRAY_JOB_ID}/task-${SLURM_ARRAY_TASK_ID}"
' EXIT ERR SIGTERM

# ---------------------------------------------------------------------------
# Stagger start slightly to avoid thundering-herd on the DB socket
# ---------------------------------------------------------------------------
sleep $(( SLURM_ARRAY_TASK_ID % 20 ))

cd "${PROJECT_DIR}"

# ---------------------------------------------------------------------------
# Wait until devenv is ready (Task 0 of the coordinator already built it)
# ---------------------------------------------------------------------------
echo "[$(date)] Worker ${SLURM_ARRAY_TASK_ID}: checking devenv..."
nix develop ./slurm --no-pure-eval --command python --version

# ---------------------------------------------------------------------------
# Verify the PostgreSQL socket is reachable before starting expensive work
# ---------------------------------------------------------------------------
STORAGE_URL="postgresql+psycopg2:///optuna_db?host=${PG_SOCKET_DIR}"

echo "[$(date)] Worker ${SLURM_ARRAY_TASK_ID}: verifying DB connection..."
nix develop ./slurm --no-pure-eval --command python - <<PYEOF
import psycopg2, sys
try:
    conn = psycopg2.connect(dbname="optuna_db", host="${PG_SOCKET_DIR}")
    conn.close()
    print("DB connection OK")
except Exception as e:
    print(f"DB connection FAILED: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Run the experiment — one trial-contributing worker
# ---------------------------------------------------------------------------
echo "[$(date)] Worker ${SLURM_ARRAY_TASK_ID}: starting experiment (optimizer=${OPTIMIZER}, dataset=${DATASET})"

TIMEOUT_SECONDS=$(python3 -c "print(int(float('${TIMEOUT_HOURS}') * 3600))")

nix develop ./slurm --no-pure-eval --command \
    env \
        PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" \
        OPTUNA_STORAGE="${STORAGE_URL}" \
        OPTUNA_STUDY_NAME="${STUDY_NAME}" \
        WORKER_TIMEOUT="${TIMEOUT_SECONDS}" \
    python "${PROJECT_DIR}/${EXPERIMENT}" \
        -p "${DATASET}" \
        -j "${SLURM_ARRAY_JOB_ID}" \
        -o "${OPTIMIZER}" \
        --worker-id "${SLURM_ARRAY_TASK_ID}"

echo "[$(date)] Worker ${SLURM_ARRAY_TASK_ID}: finished."
