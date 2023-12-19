from logging_output_scripts.utils import check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler


"""
Uses seaborn-package to create violin-Plots comparing model performances
on multiple datasets
"""
sns.set_style("whitegrid")
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_theme(style="whitegrid",
              font="Times New Roman",
              font_scale=1,
              rc={
                  "lines.linewidth": 1,
                  "pdf.fonttype": 42,
                  "ps.fonttype": 42
              })

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.tight_layout()


def create_plots(metricname = 'test_neg_mean_squared_error'):
    metric = 'metrics.' + metricname
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"

    output_dir = "violin_plots"
    check_and_create_dir(final_output_dir, output_dir)

    final_output_dir = f"{config['output_directory']}"
    scaler = MinMaxScaler()
    
    for problem in config['datasets']:
        first = True
        res_var = 0
        counter = 0
        for heuristic, renamed_heuristic in config['heuristics'].items():
            fold_df = get_df(heuristic, problem)
            if fold_df is not None:
                counter += 1
                name = [renamed_heuristic] * fold_df.shape[0]
                current_res = fold_df.assign(Used_Representation=name)
                if first:
                    first = False
                    res_var = current_res
                else:
                    # Adds additional column for plotting
                    res_var = pd.concat([res_var, current_res])

                print(f"Done for {problem} with {renamed_heuristic}")

        if counter and config["normalize_datasets"]:
            reshaped_var = np.array(res_var[metric])[-counter*64:].reshape(counter, -1) * -1
            scaler.fit(reshaped_var)
            scaled_var = scaler.transform(reshaped_var)
            scaled_var = scaled_var.reshape(1, -1)[0]
            res_var[metric][-len(scaled_var):] = scaled_var

        # Invert values since they are stored as negatives
        if not config["normalize_datasets"]:
            if metric == 'metrics.test_neg_mean_squared_error':
                res_var[metric] *= -1

        def ax_config(axis):
            ax.set_xlabel('Optimierer')
            if metric == 'metrics.test_neg_mean_squared_error':
                ax.set_ylabel('MSE')
            else:
                ax.set_ylabel('Komplexitaet')
            ax.set_box_aspect(1)

        problem = problem if not config["normalize_datasets"] else "normalized"

        # Store violin-plots of all models in one plot
        fig, ax = plt.subplots()
        ax = sns.violinplot(x='Used_Representation', y=metric, data=res_var, density_norm="width", hue='Used_Representation')
        ax_config(ax)
        fig.savefig(f"{final_output_dir}/{output_dir}/{problem}_{metricname}.png")


if __name__ == '__main__':
    create_plots()
