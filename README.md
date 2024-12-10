# SupRB experimentation

This repo contains a collection of preset experiments going beyond the examples in the main SupRB repository (https://github.com/heidmic/suprb).
The main repository contains the source code of the supervised machine learning system and references the associated publications.
This repo also features scripts to automatically perform hyperparametersearch, run on SLURM-based server infrastructure and to generate tables and graphs based on run results.


## Create an experiment on the OC server:


1) Clone repository to /data/oc-compute03/${USER}/ (if you don't have that folder already create it) 
2) Create an output directory within your cloned repository (/data/oc-compute03/${USER}/<repo_name>/output)
3) Check (and recheck) that your sbatch file is correct (see slurm/default.sbatch). Make sure that JOB_DIR and EXPERIMENT are correct and that you link to the correct path where your nix file is located
4) Call sbatch script with: sbatch slurm/default.sbatch 
5) You can check if your job is running with **squeue**. If it's not listed there, check the error in **output/output<job_id>**


## Create an experiment on licca:


Follow the same steps as above, but clone the repository on licca.


## Run logging_output_scripts

The easiest (and most comfortable) to run the scrirpts to create the graphs in logging_output_scripts is to use the **one_time_run.py**. Here is how to use it:

1) Convert your mlruns to a csv file (the scripts also work with mlruns, but the graph creation with csv is a lot faster)
    1.1) Make sure the mlruns you want to convert are in the mlruns folder from where you call **one_time_run.py**
    1.2) Use **mlruns_to_csv(datasets, subdir, normalize)** to convert to a csv file
        1.2.1) **datasets** is a dict (at the top of **one_time_run.py** there are a couple of examples)
        1.2.2) **subdir** is the directory you want to save your csv files to (they will be saved in the directory **mlruns_csv/subdir**)
        1.2.3) **normalize** is a bool to specify if you want the datasets to be normalized
2) Create a list similar to the ones already present after **if __name__ == '__main__':**
    2.1) **list[0]**: Specifies where the created graphs shall be saved
    2.2) **list[1]**: Specifies the heuristics used (it's a dict -> see the examples already present)
    2.3) **list[2]**: Specifies the axis description
    2.4) **list[3]**: Specifies if the data is normalized
    2.5) **list[3]**: Specifies the directory where the csv_mlruns are located
3) Assign **setting** to the **list** created in 2)
4) Call **run_main()**, this will create swarm plots and calvo plots in the given directory
5) For ttests graph creation, follow these steps:
    5.1) Make a new **if setting[0] == "<list[0] from 2.1>"** (similar to the ones already present)
    5.2) Call **ttest()** with the following parameters:
        5.2.1) **latex** whether you want to have a latex output in the console as well
        5.2.2) **cand1** what heuristic shall be used as candidate 1 for the ttest
        5.2.3) **cand2** what heuristic shall be used as candidate 2 for the ttest
        5.2.3) **cand1_name** what label shall be used for candidate 1
        5.2.4) **cand2_name** what label shall be used for candidate 2


Currently the file holds multiple graph creation configs, which can be commented or uncommented at the end of the file. For now it's best to keep it like this to one it can be used as examples on how to run the script and to the other, we might be needing them again.
