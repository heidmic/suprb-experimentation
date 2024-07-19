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


mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"


def create_plots():
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"

    for output_dir in ["violin_plots", "swarm_plots", "line_plots"]:
        check_and_create_dir(final_output_dir, output_dir)

    final_output_dir = f"{config['output_directory']}"
    scaler = MinMaxScaler()
    
    for problem in config['datasets']:
        first = True
        res_var = 0
        counter = 0
        fold_df = None

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

        if counter and config["normalize_datasets"]:
            reshaped_var = np.array(res_var[mse])[-counter*64:].reshape(counter, -1) * -1
            scaler.fit(reshaped_var)
            scaled_var = scaler.transform(reshaped_var)
            scaled_var = scaled_var.reshape(1, -1)[0]
            res_var[mse][-len(scaled_var):] = scaled_var

        # Invert values since they are stored as negatives
        if not config["normalize_datasets"]:
            res_var[mse] *= -1

        def ax_config(axis, y_label):
            ax.set_xlabel('Estimator', weight="bold")
            ax.set_ylabel(y_label, weight="bold")
            ax.set_title(config['datasets'][problem] if not config["normalize_datasets"]
                        else "Normalized Datasets", style="italic")
            # ax.set_box_aspect(1)

        problem = problem if not config["normalize_datasets"] else "normalized"

        ################### MSE ###########################
        plots = {"violin_plots": sns.violinplot,
                 "swarm_plots": sns.swarmplot}
        y_axis_label = {"MSE": mse,
                        "Complexity": complexity}
        
        for name, function in plots.items():
            for y_label, y_axis in y_axis_label.items():
                fig, ax = plt.subplots()
                ax = function(x='Used_Representation', y=y_axis, data=res_var, size=3)
                ax_config(ax, y_label)
                fig.savefig(f"{final_output_dir}/{name}/{problem}_{y_label}.png")
                plt.close(fig)

        # # Store line-box-plots
        # fig, ax = plt.subplots()

        # order = np.sort(res_var['Used_Representation'].unique())
        # ax = sns.boxplot(x='Used_Representation', y=mse, order=order,
        #                 showfliers=True, linewidth=0.8, showmeans=True, data=res_var)
        # ax = sns.pointplot(x='Used_Representation', y=mse, order=order,
        #                 data=res_var, ci=None, color='black')
        # ax_config(ax, 'MSE')
        # fig.savefig(f"{final_output_dir}/line_plots/{problem}.png")


if __name__ == '__main__':
    create_plots()
