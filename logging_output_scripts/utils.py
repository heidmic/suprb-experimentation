import os
import json
import mlflow


def get_all_runs():
    print("Get all mlflow runs...")

    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    all_runs_list = []
    heuristics = [key for key in config['heuristics']]
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash' and item != '0']

    for run in all_runs:
        run_name = mlflow.search_runs([run])['tags.mlflow.runName'][0]
        if any(substring in run_name for substring in heuristics):
            all_runs_list.append(mlflow.search_runs([run]))
            print(run_name)
    #     if len(all_runs_list) == len(heuristics):
    #         return all_runs_list

    # print("Something is missing")
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
            df = df[df['tags.fold'].str.contains("True", na=False)]
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
