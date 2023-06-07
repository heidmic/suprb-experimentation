# SupRB experimentation

This repo contains a collection of preset experiments going beyond the examples in the main SupRB repository (https://github.com/heidmic/suprb).
The main repository contains the source code of the supervised machine learning system and references the associated publications.
This repo also features scripts to automatically perform hyperparametersearch, run on SLURM-based server infrastructure and to generate tables and graphs based on run results.


## Create an experiment on the server:


1) Clone repository to /data/oc-compute03/${USER}/ (if you don't have that folder already create it)
2) Create an output directory within your cloned repository (/data/oc-compute03/${USER}/<repo_name>/output)
3) Check (and recheck) that your sbatch file is correct (see slurm/default.sbatch). Make sure that JOB_DIR and EXPERIMENT are correct and that you link to the correct path where your nix file is located
4) Call sbatch script with: sbatch slurm/default.sbatch 
5) You can check if your job is running with **squeue**. If it's not listed there, check the error in **output/output<job_id>**


## Run logging_output_scripts

You can run all scripts at once or run individual scripts with either (from mlflow) exported csv-files or by direclty using mlflow runs.

### Run all scripts at once

1) Adjust the **config.json** to fit your experiments:

{
    "filetype": "csv",                                      # You can choose between **csv** and **mlflow**
    "data_directory": "run_csvs",                           # Specify the directory where the data is present
    "output_directory": "logging_output_scripts/outputs",   # Specify the output_directory
    "heuristics": [                                         
        "ES",                                        # Heuristics and datasets are used to find the 
        "RS",                                        # respective heuristics/datasets, so make sure that the
        "NS",                                        # names here **only** correspond to one heuristic/dataset
        "MCNS",
        "NSLC"
    ],
    "datasets": [
        "concrete_strength",
        "combined_cycle_power_plant",
        "airfoil_self_noise",
        "energy_cool"
    ]
}

If the scripts are executed in **csv** format, the **data_directory** has to be in this format:

data_directory --- ES ------ concrete_strength
            |       |
            |       | ------ combined_cycle_power_plant
            |       |
            |       | ------ airfoil_self_noise
            |       |
            |       
            |
            |----- RS
            |
            |----- NS
            |
            |----- MCNS
            |
            |----- NSLC

2) Run the python script: **/run_all_scripts.py**
3) The specified **output_directory** will create a subdirectory for each scripts results
            

### Run a single script

1) Specify the **config.json** as above (and have the data prepared in the correct format if you use csvs)
2) Run the respective script in **logging_output_scripts**
3) The specified **output_directory** will create a subdirectory for each scripts results

Attention: **latex_tabulars** needs to specify a **summary_csv_dir** at the top of its file. (Not necessary if all scripts are executed at once) 

