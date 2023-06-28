import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_output_scripts.utils import get_dataframe, create_output_dir, config


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
create_output_dir(config['output_directory'])
create_output_dir(final_output_dir)
metric = "metrics.test_neg_mean_squared_error"


def create_plots():
    for problem in config['datasets']:
        first = True
        for heuristic in config['heuristics']:
            fold_df = get_dataframe(heuristic['name'], problem)
            if fold_df is not None:
                name = [heuristic['rename']] * fold_df.shape[0]
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
        title_dict = {"concrete_strength": "Concrete Strength",
                      "combined_cycle_power_plant": "Combined Cycle Power Plant",
                      "airfoil_self_noise": "Airfoil Self Noise",
                      "energy_cool": "Energy Efficiency Cooling"}
        ax.set_title(title_dict[problem], style="italic")
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
    for dir in ["", "violin_plots", "swarm_plots"]:
        directory = f"{final_output_dir}/{dir}"
        if not os.path.isdir(directory):
            os.mkdir(directory)

    create_plots()
