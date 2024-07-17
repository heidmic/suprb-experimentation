import os
import json
import mlflow

results_dict = {}


def filter_runs():
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    all_runs_df = mlflow.search_runs(search_all_experiments=True)

    for heuristic in config["heuristics"].keys():
        for dataset in config["datasets"].keys():
            filtered_df = all_runs_df[
                all_runs_df["tags.mlflow.runName"].str.contains(heuristic, case=False, na=False) &
                all_runs_df["tags.mlflow.runName"].str.contains(dataset, case=False, na=False) &
                (all_runs_df["tags.fold"] == 'True')
            ]
            
            if not filtered_df.empty:
                print(f"Dataframe found for {heuristic} and {dataset}")
            else:
                print(f"No run found with {heuristic} and {dataset}")

            results_dict[(heuristic, dataset)] = filtered_df

def get_df(heuristic, dataset):
    return results_dict[(heuristic, dataset)]

    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    # all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash']

    df = mlflow.search_runs(
        filter_string=f"tags.mlflow.runName ILIKE '%{heuristic}%' AND tags.mlflow.runName ILIKE '%{dataset}%' AND tags.fold = 'True'",
        search_all_experiments=True
    )

    if not df.empty:
        print(f"Dataframe found for {heuristic} and {dataset}")
        return df
    else:
        print(f"No run found with {heuristic} and {dataset}")
        exit()


    for run in all_runs:
        df = mlflow.search_runs([run])
        if not 'tags.mlflow.runName' in df:
            continue 
        print(df['tags.mlflow.runName'])
        exit()
        print(df['tags.mlflow.runName'][0])

        dataset_mask =  df['tags.mlflow.runName'].str.contains(f"{dataset}")
        fold_mask = df['tags.fold'].str.contains("True", na=False)
        
        if heuristic:
            heuristic_mask = df['tags.mlflow.runName'].str.contains(f"{heuristic}")
            df = df[heuristic_mask & dataset_mask & fold_mask]
        else:
            df = df[dataset_mask & fold_mask]
            

        if not df.empty:
            print("found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1\n\n\n\n")
            return df
    
    print(f"No run found with {heuristic} and {dataset}")


def get_all_runs(problem):
    print("Get all mlflow runs...")

    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    all_runs_list = []
    heuristics = [key for key in config['heuristics']]
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash' and item != '0']

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
