#!/usr/bin/env bash
# Run inside zellij on oc-head:
#   zellij
#   bash slurm_better/coordinator.sh --optimizer spea2 --dataset combined_cycle_power_plant --n-workers 25 --node oc-compute05 --project-dir /home/wolfbene/run1/suprb-experimentation
set -e

OPTIMIZER="spea2"
DATASET="airfoil_self_noise"
N_WORKERS=25
TUNING_TIMEOUT_HOURS=24
NODE="oc-compute04"
PROJECT_DIR="/home/$USER/suprb-experimentation"   # default

while [[ $# -gt 0 ]]; do
    case "$1" in
        --optimizer)      OPTIMIZER="$2";          shift 2 ;;
        --dataset)        DATASET="$2";             shift 2 ;;
        --n-workers)      N_WORKERS="$2";            shift 2 ;;
        --tuning-timeout) TUNING_TIMEOUT_HOURS="$2"; shift 2 ;;
        --node)           NODE="$2";                 shift 2 ;;
        --project-dir)    PROJECT_DIR="$2";          shift 2 ;;   # <-- new
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source /projects/shared/lib/slurm-utils.sh

cd "${PROJECT_DIR}"

echo "[$(date)] Warming Nix cache..."
nix develop ./slurm_better --no-pure-eval --command true

RUN_ID="${OPTIMIZER}_${DATASET}"
STUDY_NAME="${OPTIMIZER}_tuning_${DATASET}"

PG_HOST="localhost"   # DB runs on same node as workers

# Shared directory for flags (must be on network storage so coordinator can see it)
PG_BASE="${PROJECT_DIR}/.postgres_shared/${RUN_ID}"
READY_FLAG="${PG_BASE}/.pg_ready"
STOP_FLAG="${PG_BASE}/.pg_stop"
mkdir -p "$PG_BASE"

# Submit the DB job and wait for it to become ready
echo "[$(date)] Run ${RUN_ID}: submitting PostgreSQL job on ${NODE}..."
PG_JOB_ID=$(sbatch --parsable --nodelist="${NODE}" \
    --export=NONE,RUN_ID="${RUN_ID}",STUDY_NAME="${STUDY_NAME}",PG_BASE="${PG_BASE}",PROJECT_DIR="${PROJECT_DIR}" \
    slurm_better/postgres.sbatch)
echo "[$(date)] Postgres job ${PG_JOB_ID} submitted, waiting for it to become ready..."

until [ -f "$READY_FLAG" ]; do
    if ! squeue -j "${PG_JOB_ID}" -h | grep -q .; then
        echo "[$(date)] ERROR: postgres job ${PG_JOB_ID} ended before becoming ready. Check output/postgres_${PG_JOB_ID}.err" >&2
        exit 1
    fi
    sleep 10
done

PG_PORT=$(cat "${PG_BASE}/.pg_port")
echo "[$(date)] PostgreSQL ready on port ${PG_PORT}."

# Submit tuning workers
echo "[$(date)] Submitting ${N_WORKERS} tuning workers on ${NODE}..."
TUNE_ARRAY_ID=$(sbatch --parsable --nodelist="${NODE}" \
    --array=0-$((N_WORKERS - 1)) \
    --export=ALL,OPTIMIZER="${OPTIMIZER}",DATASET="${DATASET}",PG_HOST="${PG_HOST}",PG_PORT="${PG_PORT}",STUDY_NAME="${STUDY_NAME}",TIMEOUT_HOURS="${TUNING_TIMEOUT_HOURS}",PROJECT_DIR="${PROJECT_DIR}" \
    slurm_better/tuning_worker.sbatch)
echo "[$(date)] Tuning array: ${TUNE_ARRAY_ID}"

while squeue -j "${TUNE_ARRAY_ID}" -h | grep -q .; do
    sleep 60
done
echo "[$(date)] Tuning stage finished."

# Submit evaluation workers as a throttled array job
echo "[$(date)] Submitting ${N_WORKERS} evaluation workers on ${NODE}..."
EVAL_ARRAY_ID=$(sbatch --parsable --nodelist="${NODE}" \
    --array=0-$((N_WORKERS - 1)) \
    --export=ALL,OPTIMIZER="${OPTIMIZER}",DATASET="${DATASET}",PG_HOST="${PG_HOST}",PG_PORT="${PG_PORT}",STUDY_NAME="${STUDY_NAME}",PROJECT_DIR="${PROJECT_DIR}" \
    slurm_better/eval_worker.sbatch)
echo "[$(date)] Evaluation array: ${EVAL_ARRAY_ID}"

while squeue -j "${EVAL_ARRAY_ID}" -h | grep -q .; do
    sleep 60
done
echo "[$(date)] Evaluation stage finished."


echo "[$(date)] Merging per-worker mlruns into shared store..."
MLRUNS_EVAL_DIR="${PROJECT_DIR}/mlruns-eval-${EVAL_ARRAY_ID}"
nix develop ./slurm_better --no-pure-eval --command python \
    "${PROJECT_DIR}/slurm_better/merge_mlruns.py" \
    --dest "file://${PROJECT_DIR}/mlruns" \
    --sources ${MLRUNS_EVAL_DIR}/task-*
echo "[$(date)] mlruns merge complete."

echo "[$(date)] Signalling postgres job to export + shut down..."
touch "$STOP_FLAG"

while squeue -j "${PG_JOB_ID}" -h | grep -q .; do
    sleep 15
done
echo "[$(date)] Run ${RUN_ID} complete. Study exported to studies/${STUDY_NAME}.db"