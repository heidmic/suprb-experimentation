import os
import json
import mlflow
import pandas as pd

with open('logging_output_scripts/config.json') as f:
    config = json.load(f)


if config['filetype'] == 'mlflow':
    print("Get all mlflow runs...")
    all_runs_list = []
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash' and item != '0']
    for run in all_runs:
        all_runs_list.append(mlflow.search_runs([run]))


def get_dataframe(heuristic, problem):
    if config['filetype'] == 'csv':
        return get_csv_dataframe(heuristic, problem)
    elif config['filetype'] == 'mlflow':
        return get_mlflow_dataframe(heuristic, problem)


def get_mlflow_dataframe(heuristic, dataset):
    df = None

    for run in all_runs_list:
        df = run[run['tags.mlflow.runName'].str.contains(f" {heuristic} ")]

        if not df.empty:
            df = df[df['tags.mlflow.runName'].str.contains(dataset)]
        else:
            continue

        if not df.empty:
            # Filter out individual runs (Removes averaged values)
            df = df[df['tags.fold'].str.contains("True", na=False)]
            return df
        else:
            continue

    return None


def get_csv_dataframe(heuristic, problem):
    df = pd.read_csv(f"{config['data_directory']}/{heuristic}/{problem}.csv")
    return df[df['Name'].str.contains('fold')]


def create_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
