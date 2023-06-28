import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from logging_output_scripts.utils import get_dataframe, check_and_create_dir, config, get_all_runs


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


final_output_dir = f"{config['output_directory']}"
metric = "metrics.test_neg_mean_squared_error"


def create_plots():
    all_runs_list = get_all_runs()
    for problem in config['datasets']:
        first = True
        res_var = 0
        for heuristic, renamed_heuristic in config['heuristics'].items():
            fold_df = get_dataframe(all_runs_list, heuristic, problem)
            if fold_df is not None:
                name = [renamed_heuristic] * fold_df.shape[0]
                current_res = fold_df.assign(Used_Representation=name)
                if first:
                    first = False
                    res_var = current_res
                else:
                    # Adds additional column for plotting
                    res_var = pd.concat([res_var, current_res])

                print(f"Done for {problem} with {heuristic}")

    # Invert values since they are stored as negatives
    res_var[metric] *= -1

    def ax_config(axis):
        ax.set_xlabel('Estimator', weight="bold")
        ax.set_ylabel('MSE', weight="bold")
        ax.set_title(config['datasets'][problem], style="italic")
        ax.set_box_aspect(1)

    # Store violin-plots of all models in one plot
    fig, ax = plt.subplots()
    ax = sns.violinplot(x='Used_Representation', y=metric, data=res_var, scale="width", scale_hue=False)
    ax_config(ax)
    fig.savefig(f"{final_output_dir}/violin_plots/{problem}.png")

    # Store swarm-plots of all models in one plot
    fig, ax = plt.subplots()
    ax = sns.swarmplot(x='Used_Representation', y=metric, data=res_var, size=2)
    ax_config(ax)
    fig.savefig(f"{final_output_dir}/swarm_plots/{problem}.png", dpi=500)


if __name__ == '__main__':
    for output_dir in ["", "violin_plots", "swarm_plots"]:
        check_and_create_dir(final_output_dir, output_dir)

    create_plots()
