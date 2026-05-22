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