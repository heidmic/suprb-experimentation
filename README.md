# SupRB experimentation

This repo contains a collection of preset experiments going beyond the examples in the main SupRB repository (https://github.com/heidmic/suprb).
The main repository contains the source code of the supervised machine learning system and references the associated publications.
This repo also features scripts to automatically perform hyperparametersearch, run on SLURM-based server infrastructure and to generate tables and graphs based on run results.


## Create an experiment on the server:


1) Clone repository to /data/oc-compute03/${USER}/ (if you don't have that folder already create it)
2) Create an output directory within your cloned repository (/data/oc-compute03/${USER}/<repo_name>/output)
3) Check (and recheck) that your sbatch file is correct (see slurm/default.sbatch). Make sure that JOB_DIR and EXPERIMENT are correct and that you link to the correct path where your nix file is located
4) Call sbatch script with: sbatch slurm/default.sbatch 

