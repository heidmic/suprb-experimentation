#!/usr/bin/env bash
###############################################################################
# Slurm directives 
###############################################################################
#SBATCH --time=96:00:00                 # 96h wall‑clock limit
#SBATCH --partition=cpu                 # Passende Partition für compute05
#SBATCH --nodelist=oc-compute05         # Ziel-Node auf compute05 geändert
#SBATCH --cpus-per-task=5              # 5 CPUs per array task
#SBATCH --mem=20G                      # 20 GiB RAM per array task
#SBATCH --output=output/%A_%a.out      # stdout file
#SBATCH --error=output/%A_%a.err       # stderr file
#
# Submit: sbatch slurm/jobsingle.sbatch
###############################################################################

source /projects/shared/lib/slurm-utils.sh


trap 'echo "[$(date)] Trap ausgelöst..."; save_results "$SCRATCH" ~"/results/" 

PROJECT_DIR="/home/$USER/suprb-experimentation"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"
EXPERIMENT="runs/comparisons/suprb_tuning.py"


cd "${PROJECT_DIR}"


echo "[$(date)] Task baut die Devenv-Umgebung exklusiv vor..."
    # Initialisiert die Umgebung einmal komplett isoliert
nix develop ./slurm --no-pure-eval --command python --version


# Jetzt rufen wir den Python-Befehl auf
nix develop ./slurm --no-pure-eval --command env PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}" python "${PROJECT_DIR}/${EXPERIMENT}" \
    -p "${dataset}" \
    -j "${SLURM_ARRAY_JOB_ID}" \
    -o "${OPTIMIZER}"

