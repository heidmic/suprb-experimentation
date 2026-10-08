#!/usr/bin/env bash
# Run inside zellij on oc-head:
#   zellij
#   bash slurm_better/eval_coordinator.sh --optimizer spea2 --dataset airfoil_self_noise --n-workers 25 --node oc-compute04 --project-dir /home/wolfbene/thesis/suprb-experimentation
set -e



# ----------------------------
# Default parameters
# ----------------------------
OPTIMIZER="spea2"
DATASET="airfoil_self_noise"
N_WORKERS=25
NODE="oc-compute04"
PROJECT_DIR="/home/$USER/thesis/asn_32_4"   # change to your project root

# ----------------------------
# Parse command line arguments
# ----------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --optimizer)      OPTIMIZER="$2";          shift 2 ;;
        --dataset)        DATASET="$2";             shift 2 ;;
        --n-workers)      N_WORKERS="$2";            shift 2 ;;
        --node)           NODE="$2";                 shift 2 ;;
        --project-dir)    PROJECT_DIR="$2";          shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ----------------------------
# Source Slurm utilities (defines save_results, SCRATCH, etc.)
# ----------------------------
source /projects/shared/lib/slurm-utils.sh

# ----------------------------
# Go to the project directory
# ----------------------------
cd "${PROJECT_DIR}"

# ----------------------------
# Define study name and SQLite DB path
# ----------------------------

STUDY_NAME="${OPTIMIZER}_tuning_${DATASET}"
STUDY_DB="${PROJECT_DIR}/studies/${STUDY_NAME}.db"
OPTUNA_STORAGE="sqlite:///${STUDY_DB}"

# Check if the database exists (optional but helpful)
if [ ! -f "${STUDY_DB}" ]; then
    echo "[$(date)] WARNING: Study DB not found at ${STUDY_DB}" >&2
    echo "          Continuing anyway, but ensure it will be created or already exists." >&2
fi

# ----------------------------
#  warm Nix cache 
# ----------------------------
echo "[$(date)] Warming Nix cache..."
nix develop ./slurm_better --no-pure-eval --command true

# ----------------------------
# Submit evaluation array job
# ----------------------------
echo "[$(date)] Submitting ${N_WORKERS} evaluation workers on ${NODE}..."
EVAL_ARRAY_ID=$(sbatch --parsable --nodelist="${NODE}" \
    --array=0-$((N_WORKERS - 1)) \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",OPTIMIZER="${OPTIMIZER}",DATASET="${DATASET}",STUDY_NAME="${STUDY_NAME}",OPTUNA_STORAGE="${OPTUNA_STORAGE}" \
    slurm_better/eval_worker.sbatch)

echo "[$(date)] Evaluation array job ID: ${EVAL_ARRAY_ID}"

# ----------------------------
# Wait for the array to finish
# ----------------------------
echo "[$(date)] Waiting for evaluation array to complete..."
while squeue -j "${EVAL_ARRAY_ID}" -h | grep -q .; do
    sleep 60
done
echo "[$(date)] Evaluation array finished."

# ----------------------------
# Merge per‑worker mlruns into a shared store
# (This mimics the coordinator's merge step)
# ----------------------------
MLRUNS_EVAL_DIR="${PROJECT_DIR}/mlruns-eval-${EVAL_ARRAY_ID}"
if [ -d "${MLRUNS_EVAL_DIR}" ]; then
    echo "[$(date)] Merging per-worker mlruns from ${MLRUNS_EVAL_DIR} into central store..."
    nix develop ./slurm_better --no-pure-eval --command python \
        "${PROJECT_DIR}/slurm_better/merge_mlruns.py" \
        --dest "file://${PROJECT_DIR}/mlruns" \
        --sources ${MLRUNS_EVAL_DIR}/task-*
    echo "[$(date)] Merge complete."
else
    echo "[$(date)] No mlruns-eval directory found; skipping merge."
fi

# ----------------------------
# Done
# ----------------------------
echo "[$(date)] Evaluation run complete. Study DB: ${STUDY_DB}"