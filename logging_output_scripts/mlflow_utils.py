import os
import mlflow


all_runs_list = []
all_runs = [item for item in next(os.walk('mlruns'))[1] if item != '.trash' and item != '0']
for run in all_runs:
    all_runs_list.append(mlflow.search_runs([run]))


def get_dataframe(heuristic, dataset):
    df = None

    for run in all_runs_list:
        df = run[run['tags.mlflow.runName'].str.contains(f" {heuristic} Tuning ")]

        if not df.empty:
            df = df[df['tags.mlflow.runName'].str.contains(dataset)]
        else:
            continue

        if not df.empty:
            # Filter out individual runs (Removes averaged values)
            df = df[df['tags.fold'].str.contains("True", na=False)]
            break
        else:
            continue

    return df
