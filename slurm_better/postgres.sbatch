#!/usr/bin/env bash
#SBATCH --job-name=study_postgres
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=120:00:00
#SBATCH --output=output/postgres_%j.out
#SBATCH --error=output/postgres_%j.err

source /projects/shared/lib/slurm-utils.sh

if [ -z "$PROJECT_DIR" ]; then
    PROJECT_DIR="/home/$USER/suprb-experimentation"
fi
cd "${PROJECT_DIR}"

if [ -n "$USER_SCRATCH" ]; then
    export XDG_RUNTIME_DIR="${USER_SCRATCH}/xdg_runtime_${SLURM_JOB_ID}_${RUN_ID}"
    mkdir -p "$XDG_RUNTIME_DIR"
    chmod 700 "$XDG_RUNTIME_DIR"
    echo "[$(date)] Overrode XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}"
else
    echo "Warning: USER_SCRATCH not set, leaving XDG_RUNTIME_DIR as-is — midnight nix develop failures may recur" >&2
fi

find_free_port() {
    local base=15000
    local max=20000
    local tries=50
    for ((i = 0; i < tries; i++)); do
        local port=$((base + RANDOM % (max - base)))
        if ! ss -tln 2>/dev/null | grep -q ":${port} "; then
            echo "$port"
            return 0
        fi
    done
    echo "ERROR: could not find a free port on $(hostname) after ${tries} tries" >&2
    return 1
}

PG_PORT=$(find_free_port) || exit 1
echo "[$(date)] Selected PG_PORT=${PG_PORT} on $(hostname)"

# Publish the chosen port so coordinator.sh can read it once ready and pass
# it on to the tuning/eval worker jobs (it's no longer set at submit time).
echo "$PG_PORT" > "${PG_BASE}/.pg_port"

if [ -n "$USER_SCRATCH" ]; then
    PG_DATA="${USER_SCRATCH}/pg_data_${SLURM_JOB_ID}_${RUN_ID}"
    PG_SOCKET_DIR="${USER_SCRATCH}/socket_${SLURM_JOB_ID}_${RUN_ID}"
else
    echo "Warning: USER_SCRATCH not set, using /tmp" >&2
    PG_DATA="/tmp/$USER/pg_data_${SLURM_JOB_ID}_${RUN_ID}"
    PG_SOCKET_DIR="/tmp/$USER/socket_${SLURM_JOB_ID}_${RUN_ID}"
fi

READY_FLAG="${PG_BASE}/.pg_ready"
STOP_FLAG="${PG_BASE}/.pg_stop"

SQLITE_DIR="${PROJECT_DIR}/studies"
SQLITE_PATH="${SQLITE_DIR}/${STUDY_NAME}.db"

mkdir -p "$PG_DATA" "$PG_SOCKET_DIR" "$SQLITE_DIR"
rm -f "$STOP_FLAG" "$READY_FLAG"

STORAGE_URL="postgresql://localhost:${PG_PORT}/optuna_db"

echo "[$(date)] Checking for stale orphaned postgres instances for ${STUDY_NAME}..."
for pid in $(pgrep -f "postgres.*pg_data_.*_${STUDY_NAME}"); do
    cmdline=$(ps -o args= -p "$pid" 2>/dev/null)
    old_job_id=$(echo "$cmdline" | grep -oP 'pg_data_\K[0-9]+' | head -1)
    if [ -n "$old_job_id" ] && [ "$old_job_id" != "$SLURM_JOB_ID" ]; then
        if ! squeue -h -j "$old_job_id" &>/dev/null || [ -z "$(squeue -h -j "$old_job_id" 2>/dev/null)" ]; then
            echo "[$(date)] Found orphaned postgres (pid $pid) from dead job $old_job_id — killing it"
            kill "$pid" 2>/dev/null
            sleep 2
            kill -0 "$pid" 2>/dev/null && { echo "[$(date)] pid $pid still alive, force-killing"; kill -9 "$pid" 2>/dev/null; }
        fi
    fi
done

# Cleanup: stop PostgreSQL and remove the data directories from USER_SCRATCH
cleanup() {

    if [ -f "$READY_FLAG" ]; then
        echo "[$(date)] Exporting study to SQLite..."
        nix develop ./slurm_better --no-pure-eval --command python -u - <<PYEOF 2>&1 | tee -a "${HOME}/export_${SLURM_JOB_ID}.log"
import optuna
try:
    optuna.copy_study(from_study_name="${STUDY_NAME}", from_storage="${STORAGE_URL}", to_storage="sqlite:///${SQLITE_PATH}")
    print("Study exported to ${SQLITE_PATH}")
except Exception as e:
    print(f"SQLite export failed: {e}")
PYEOF
    fi

    echo "[$(date)] Cleanup: stopping PostgreSQL..."
    if ! nix develop ./slurm_better --no-pure-eval --command pg_ctl -D "$PG_DATA" stop -m fast; then
        echo "[$(date)] WARNING: pg_ctl stop failed or server was already down — check for a lingering process" >&2
    fi

    

    rm -rf "$PG_DATA" "$PG_SOCKET_DIR" "$XDG_RUNTIME_DIR"
    rm -f "$READY_FLAG" "$STOP_FLAG" "${PG_BASE}/.pg_port"
}
trap cleanup EXIT

if [ ! -f "$PG_DATA/PG_VERSION" ]; then
    echo "[$(date)] Initialising PostgreSQL data directory..."
    nix develop ./slurm_better --no-pure-eval --command initdb -D "$PG_DATA" --no-locale --auth=trust --no-sync
fi

echo "[$(date)] Starting PostgreSQL on port ${PG_PORT}..."
nix develop ./slurm_better --no-pure-eval --command \
    pg_ctl -D "$PG_DATA" -o "-h localhost -p ${PG_PORT} -N 200 -c dynamic_shared_memory_type=sysv -c unix_socket_directories=${PG_SOCKET_DIR} -c fsync=off -c synchronous_commit=off -c full_page_writes=off -c checkpoint_timeout=15min -c max_wal_size=2GB -c shared_buffers=512MB" \
    -l "${PG_BASE}/postgres.log" start

if [ $? -ne 0 ]; then
    echo "[$(date)] FATAL: pg_ctl start failed — check ${PG_BASE}/postgres.log. Aborting rather than risk talking to an unrelated server." >&2
    exit 1
fi

echo "[$(date)] Waiting for PostgreSQL to accept connections on port ${PG_PORT}..."
until nix develop ./slurm_better --no-pure-eval --command pg_isready -h localhost -p "$PG_PORT"; do
    sleep 1
    echo "[$(date)] checking..."
done
echo "[$(date)] PostgreSQL is ready."

actual_data_dir=$(nix develop ./slurm_better --no-pure-eval --command \
    psql -h localhost -p "$PG_PORT" -d postgres -tAc "SHOW data_directory;" 2>/dev/null)
if [ "$actual_data_dir" != "$PG_DATA" ]; then
    echo "[$(date)] FATAL: server on port ${PG_PORT} reports data_directory='${actual_data_dir}', expected '${PG_DATA}'. Refusing to proceed against an unexpected instance." >&2
    exit 1
fi
echo "[$(date)] Confirmed server identity matches our own data directory."

echo "[$(date)] Creating database optuna_db..."
nix develop ./slurm_better --no-pure-eval --command createdb -h localhost -p "$PG_PORT" optuna_db

echo "[$(date)] Creating Optuna study..."
nix develop ./slurm_better --no-pure-eval --command python -u - <<PYEOF 2>&1 | tee "${HOME}/create_study_${SLURM_JOB_ID}.log"
import optuna
study = optuna.create_study(study_name="${STUDY_NAME}", storage="${STORAGE_URL}", load_if_exists=True)
print(f"Study created/loaded: {study.study_name}")
PYEOF

if [ $? -ne 0 ]; then
    echo "[$(date)] ERROR: Study creation failed. Check ${HOME}/create_study_${SLURM_JOB_ID}.log" >&2
    exit 1
fi

echo "[$(date)] Touching ready flag..."
touch "$READY_FLAG"



while [ ! -f "$STOP_FLAG" ]; do
    
    sleep 10
done