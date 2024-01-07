import os
import json
import mlflow

def get_df(heuristic, dataset):
    with open('config.json') as f:
        config = json.load(f)

    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash']

    for run in all_runs:
        df = mlflow.search_runs([run])
        print(df['tags.mlflow.runName'])

        heuristic_mask = df['tags.mlflow.runName'].str.contains(f"{heuristic}")
        dataset_mask =  df['tags.mlflow.runName'].str.contains(f"{dataset}")
        #fold_mask = df['tags.fold'].str.contains("True", na=False)
        #df = df[heuristic_mask & dataset_mask & fold_mask]
        df = df[heuristic_mask & dataset_mask]

        if not df.empty:
            return df

    print(f"No run found with {heuristic} and {dataset}")


def get_all_runs(problem):
    print("Get all mlflow runs...")

    with open('config.json') as f:
        config = json.load(f)

    all_runs_list = []
    heuristics = [key for key in config['heuristics']]
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash']

    for run in all_runs:
        for run_name in mlflow.search_runs([run])['tags.mlflow.runName']:
            if problem in run_name:
                if any(substring in run_name for substring in heuristics):
                    all_runs_list.append(mlflow.search_runs([run]))
                    print(run_name)

    return all_runs_list


def get_dataframe(all_runs_list, heuristic, dataset):
    df = None
    for run in all_runs_list:
        df = run[run['tags.mlflow.runName'].str.contains(f"{heuristic}")]

        if not df.empty:
            df = df[df['tags.mlflow.runName'].str.contains(f"{dataset}")]
        else:
            continue

        if not df.empty:
            # Filter out individual runs (Removes averaged values)
            #df = df[df['tags.fold'].str.contains("True", na=False)]
            return df
        else:
            continue

    return None


def check_and_create_dir(output_folder, output_dir):
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    directory = f"{output_folder}/{output_dir}"

    if not os.path.isdir(directory):
        os.mkdir(directory)