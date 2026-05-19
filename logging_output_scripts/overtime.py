from logging_output_scripts.utils import check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import mlflow
import os
from sklearn.preprocessing import MinMaxScaler

"""
Uses seaborn-package to create MSE-Time and Complexity-Time Plots comparing model performances
on multiple datasets
"""
sns.set_style("whitegrid")
sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})
sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    font_scale=1,
    rc={"lines.linewidth": 1, "pdf.fonttype": 42, "ps.fonttype": 42},
)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.tight_layout()


def get_histogram(heuristic_name, dataset_name, metric_name, steps):
    with open("logging_output_scripts/config.json") as f:
        config = json.load(f)

    client = mlflow.tracking.MlflowClient()
    all_runs = [item for item in next(os.walk(config["data_directory"]))[1] if item != ".trash"]
    for run in all_runs:
        exp = client.get_experiment(run)
        if dataset_name in exp.name and heuristic_name in exp.name:

            run_ids = [item for item in next(os.walk(config["data_directory"] + "/" + str(run)))[1] if item != ".trash"]
            exp_res = []
            for i_run, id in enumerate(run_ids):
                run = client.get_run(id)
                if "fold" in run.data.tags and run.data.tags["fold"] == "True":
                    metrics = client.get_metric_history(id, metric_name)
                    run_res = np.zeros((len(metrics)))
                    for i, metric in enumerate(metrics):
                        run_res[i] = metric.value
                    exp_res.append(run_res)
            exp_res = np.average(np.array(exp_res), axis=0)
            return exp_res[:steps]


def create_plots(metric_name="elitist_complexity", steps=32):
    with open("logging_output_scripts/config.json") as f:
        config = json.load(f)
    final_output_dir = f"{config['output_directory']}"
    output_dir = "time_plots"
    check_and_create_dir(final_output_dir, output_dir)

    for dataset_name in config["datasets"]:
        results = [[], [], []]
        legend_labels = []
        for heuristic_name in config["heuristics"]:
            result = get_histogram(heuristic_name, dataset_name, metric_name, steps)
            for i, res in enumerate(result):
                results[0].append(res)
                results[1].append(i)
                results[2].append(heuristic_name)
            legend_labels.append(config["heuristics"][heuristic_name])

        results = {metric_name: results[0], "step": results[1], "optimizer_name": results[2]}
        res_data = pd.DataFrame(results)

        def ax_config(axis):
            axis.set_xlabel("Iteration")
            axis.set_ylabel(config["metrics"][metric_name])
            axis.legend(title="Optimierer", labels=legend_labels)

        fig, ax = plt.subplots()
        ax = sns.lineplot(x="step", y=metric_name, data=res_data, style="optimizer_name", hue="optimizer_name")
        ax_config(ax)
        fig.savefig(f"{final_output_dir}/{output_dir}/{dataset_name}_{metric_name}.png")


if __name__ == "__main__":
    create_plots(metric_name="elitist_error")
