import os
import json
import mlflow
import pandas as pd

results_dict = {}


def filter_runs(all_runs_df=None):
    with open('logging_output_scripts/config_regression.json') as f:
        config = json.load(f)

    if all_runs_df is not None:
        all_runs_df = mlflow.search_runs(search_all_experiments=True)

    for model in config["model_names"].keys():
        model_name = f'l:{model}'
        for dataset in config["datasets"].keys():
            filtered_df = all_runs_df[
                all_runs_df["tags.mlflow.runName"].str.contains(model_name, case=False, na=False) &
                all_runs_df["tags.mlflow.runName"].str.contains(dataset, case=False, na=False) &
                (all_runs_df["tags.fold"] == 'True')
            ]

            if not filtered_df.empty:
                print(f"Dataframe found for {model} and {dataset}")
            else:
                print(f"No run found with {model} and {dataset}")

            results_dict[(model, dataset)] = filtered_df


def get_normalized_df(exp_name):
    with open('logging_output_scripts/config_regression.json') as f:
        config = json.load(f)
    df = pd.DataFrame()
    for dataset in config["datasets"]:
        df = pd.concat([df, pd.read_csv(f"{dataset}_all.csv")])

    return df[df["tags.mlflow.runName"].str.contains(exp_name, case=False, na=False)]


def get_csv_df(exp_name, dataset):
    with open('logging_output_scripts/config_regression.json') as f:
        config = json.load(f)

    df = pd.read_csv(f"{config['data_directory']}/{dataset}_all.csv")
    return df[df["tags.mlflow.runName"].str.contains(exp_name, case=False, na=False)]


def get_df(exp_name, dataset):
    #return results_dict[(exp_name, dataset)]

    with open('logging_output_scripts/config_regression.json') as f:
        config = json.load(f)

    # all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash']

    df = mlflow.search_runs(
        filter_string=f"tags.mlflow.runName ILIKE '%{exp_name}%' AND tags.mlflow.runName ILIKE '%{dataset}%' AND tags.fold = 'True'",
        search_all_experiments=True
    )

    if not df.empty:
        print(f"Dataframe found for {exp_name} and {dataset}")
        return df
    else:
        print(f"No run found with {exp_name} and {dataset}")
        exit()

    for run in all_runs:
        df = mlflow.search_runs([run])
        if not 'tags.mlflow.runName' in df:
            continue
        print(df['tags.mlflow.runName'])
        exit()
        print(df['tags.mlflow.runName'][0])

        dataset_mask = df['tags.mlflow.runName'].str.contains(f"{dataset}")
        fold_mask = df['tags.fold'].str.contains("True", na=False)

        if exp_name:
            exp_name_mask = df['tags.mlflow.runName'].str.contains(f"{exp_name}")
            df = df[exp_name_mask & dataset_mask & fold_mask]
        else:
            df = df[dataset_mask & fold_mask]

        if not df.empty:
            print("found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n\n\n\n")
            return df

    print(f"No run found with {exp_name} and {dataset}")
    return None


def get_all():
    print("Get all mlflow runs...")
    
    with open('logging_output_scripts/config_regression.json') as f:
        config = json.load(f)
    client = mlflow.tracking.MlflowClient()
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash']
    all_runs_list = []
    for run in all_runs:
        exp = client.get_experiment(run)
        all_runs_list.append(mlflow.search_runs([run]))

    return all_runs_list


def get_all_runs(problem):
    print(f"Get all mlflow runs of {problem}...")

    with open('logging_output_scripts/config_regression.json') as f:
        config = json.load(f)

    all_runs_list = []
    models = [key for key in config['model_names']]
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash' and item != '0']

    for run in all_runs:
        for run_name in mlflow.search_runs([run])['tags.mlflow.runName']:
            if problem in run_name:
                if any(substring in run_name for substring in models):
                    all_runs_list.append(mlflow.search_runs([run]))
                    print(run_name)

    return all_runs_list

def get_by_config(config, search_string, filter_swapped=True):
    print(f"Get all mlflow runs of {search_string}...")

    all_runs_list = []
    models = [key for key in config['model_names']]
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash' and item != '0']
    for run in all_runs:
        for run_name in mlflow.search_runs([run]).get('tags.mlflow.runName', "No run name"):
            if search_string in run_name:
                if filter_swapped:
                    if 'n:' in run_name:
                        continue
                if any(string in run_name for string in models):
                    all_runs_list.append(mlflow.search_runs([run]))
    return all_runs_list


def get_dataframe(all_runs_list, exp_name, dataset):
    df = None
    for run in all_runs_list:
        df = run[run['tags.mlflow.runName'].str.contains(f"{exp_name}")]

        if not df.empty:
            df = df[df['tags.mlflow.runName'].str.contains(f"{dataset}")]
        else:
            continue

        if not df.empty:
            # Filter out individual runs (Removes averaged values)
            df = df[df['tags.fold'].str.contains("True", na=False)]
            return df
        else:
            continue

    return None


def get_all_dataframe(all_runs_list, exp_name, dataset):
    ret = None
    # Returns experiments split over multiple runs
    for run in all_runs_list:
        df = run[run['tags.mlflow.runName'].str.contains(f"{exp_name}")]
        
        if not df.empty:
            df = df[df['tags.mlflow.runName'].str.contains(f"{dataset}")]
        
        if not df.empty:
            # Filter out individual runs (Removes averaged values)
            df = df[df['tags.fold'].str.contains("True", na=False)]
            ret = pd.concat([ret, df], ignore_index=True) if ret is not None else df

    return ret


def check_and_create_dir(output_folder, output_dir):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    directory = f"{output_folder}/{output_dir}"

    if not os.path.isdir(directory):
        os.mkdir(directory)


datasets_map = {
    "combined_cycle_power_plant": "ccpp",
    "airfoil_self_noise": "asn",
    "concrete_strength": "cs",
    "energy_cool": "eec",
    "protein_structure": "pppts",
    "parkinson_total": "pt",
    "normalized": "normalized",
    "0": "0",
    "GeneticAlgorithm": "GA",
    "RandomSearch": "RS",
    "ArtificialBeeColonyAlgorithm": "ABC",
    "AntColonyOptimization": "ACO",
    "GreyWolfOptimizer": "GWO",
    "ParticleSwarmOptimization": "PSO",
    "ES Tuning": "ES",  # "SupRB", #"ES",
    "RS Tuning": "RS",
    " NS True": "NS-P",
    "MCNS True": "MCNS-P",
    "NSLC True": "NSLC-P",
    " NS False": "NS-G",
    "MCNS False": "MCNS-G",
    "NSLC False": "NSLC-G",
    "XCSF": "XCSF",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "s:ga": "GA",
    "s:saga1": "SAGA1",
    "s:saga2": "SAGA2",
    "s:saga3": "SAGA3",
    "s:sas": "SAGA4",
    "lasso": "Lasso",
    "ridge": "Ridge",
    "elasticnet": "Elastic Net",
    "elasticNet": "Elastic Net",
        "breastcancer": "Breast Cancer",
        "abalone": "Abalone",
        "raisin": "Raisin",
        "concrete_strength": "Concrete Strength",
        "airfoil_self_noise": "Airfoil Self Noise",
        "combined_cycle_power_plant": "Combined Cycle Power Plant",
        "energy_heat": "Energy Efficiency target Heat Load"
}
