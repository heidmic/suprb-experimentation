from logging_output_scripts.utils import get_csv_df, get_normalized_df, check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from utils import datasets_map
from sklearn.preprocessing import MinMaxScaler

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"


def create_plots():
    """
    Uses seaborn-package to create violin-Plots comparing model performances
    on multiple datasets
    """
    sns.set_style("whitegrid")
    sns.set_theme(style="whitegrid",
                  font="Times New Roman",
                  font_scale=1.7,
                  rc={
                      "lines.linewidth": 1,
                      "pdf.fonttype": 42,
                      "ps.fonttype": 42
                  })

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 200

    plt.tight_layout()

    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"
    scaler = MinMaxScaler()

    for problem in config['datasets']:
        first = True
        res_var = 0
        counter = 0
        fold_df = None

        for heuristic, renamed_heuristic in config['heuristics'].items():
            if config["normalize_datasets"]:
                fold_df = get_normalized_df(heuristic, "mlruns_csv/MIX")
            else:
                if config["data_directory"] == "mlruns":
                    fold_df = get_df(heuristic, problem)
                else:
                    fold_df = get_csv_df(heuristic, problem)
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

        # Invert values since they are stored as negatives
        if not config["normalize_datasets"] and config["data_directory"] == "mlruns":
            res_var[mse] *= -1

        def ax_config(axis, y_label):
            x_lab = ""
            ax.set_ylabel(y_label, weight="bold")
            ax.set_title(config['datasets'][problem], style="italic", fontsize=14)
            ax.set_xlabel(x_lab, weight="bold", labelpad=10)

            # Change this to adjust y_axis ticks
            y_min = max(0, min(ax.get_yticks()))
            y_max = min(1, max(ax.get_yticks()))

            # Change this to adjust the tick size
            num_ticks = 7

            ax.set_ylim(y_min, y_max)
            y_tick_positions = np.linspace(y_min, y_max, num_ticks)
            y_tick_positions = np.round(y_tick_positions, 3)

            plt.yticks(y_tick_positions, [f'{x:.3g}' for x in y_tick_positions])

        ################### MSE ###########################
        plots = {  # "violin": sns.violinplot,
            "swarm": sns.swarmplot,
            #  "box": sns.boxplot
        }

        y_axis_label = {"MSE": mse,
                        "Complexity": complexity
                        }

        f_index = heuristic.find('f:')
        result = heuristic[f_index+2:]

        for name, function in plots.items():
            for y_label, y_axis in y_axis_label.items():
                fig, ax = plt.subplots(dpi=400)
                plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.22)
                ax = function(x='Used_Representation', y=y_axis, data=res_var, size=3)
                ax_config(ax, y_label)

                fig.savefig(f"{final_output_dir}/{name}_{datasets_map[problem]}_{y_label}.png")
                plt.close(fig)

        if config["data_directory"] == "mlruns_csv/MIX":
            return


if __name__ == '__main__':
    create_plots()
