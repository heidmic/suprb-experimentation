from logging_output_scripts.utils import get_csv_df, get_normalized_df, check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from utils import datasets_map
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FuncFormatter
import math
from matplotlib.ticker import MaxNLocator

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"


def get_exponent(problem):
    if problem == "combined_cycle_power_plant":
        exponent = 4
    if problem == "airfoil_self_noise":
        exponent = 2
    if problem == "concrete_strength":
        exponent = 2
    if problem == "energy_cool":
        exponent = 2
    if problem == "protein_structure":
        exponent = 2
    if problem == "parkinson_total":
        exponent = 2

    return exponent


def create_formatter(exponent):
    def format_yaxis(value, _):
        return int(value * math.pow(10, exponent))

    return format_yaxis


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

        if counter and config["normalize_datasets"]:
            reshaped_var = np.array(res_var[mse])[-counter*64:].reshape(counter, -1) * -1
            scaler.fit(reshaped_var)
            scaled_var = scaler.transform(reshaped_var)
            scaled_var = scaled_var.reshape(1, -1)[0]
            res_var[mse][-len(scaled_var):] = scaled_var

        # Invert values since they are stored as negatives
        if not config["normalize_datasets"] and config["data_directory"] == "mlruns":
            res_var[mse] *= -1

        def ax_config(axis, y_label):
            x_lab = "Number of rules participating" if config["normalize_datasets"] else "RD method"
            ax.set_xlabel(x_lab, weight="bold")
            ax.set_ylabel(y_label, weight="bold")
            # ax.set_title(config['datasets'][problem] if not config["normalize_datasets"] else result, style="italic")

            if config["data_directory"] == "mlruns_csv/RD":
                label_temp = config['heuristics'].values()
                tick_positions = np.arange(len(label_temp))
                new_labels = []
                for i, label in enumerate(label_temp):
                    if i % 2 == 0:
                        new_labels.append(label + "\n")  # Nach unten verschieben mit Zeilenumbruch
                    else:
                        new_labels.append("\n" + label)  # Nach oben verschieben

                # Setzen der neuen Labels
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(new_labels)

            if y_label != "Complexity":
                num_ticks = 5  # You can adjust this value based on your needs
                y_min = min(ax.get_yticks())
                y_max = max(ax.get_yticks())
                y_tick_positions = np.linspace(y_min, y_max, num_ticks)
                # y_tick_positions = np.round(np.arange(min(ax.get_yticks()), max(ax.get_yticks()), 0.005), 3)
                y_tick_positions = np.round(y_tick_positions, 3)

                print(y_min, y_max, "   ", ax.get_yticks(), "     ",  y_tick_positions, -
                      np.log10(y_tick_positions[1] - y_tick_positions[0]))

                # Set the custom ticks and labels on the y-axis
                plt.yticks(y_tick_positions, [f'{x:.3g}' for x in y_tick_positions])

            # if y_label != "Complexity":
            #     exponent = get_exponent(problem)
            #     ax.yaxis.set_major_formatter(FuncFormatter(create_formatter(exponent)))

            #     ax.text(x=-0.05, y=0.95,  # Position of the text (x and y coordinates)
            #             s=fr"$e^{{-{exponent}}}$",  # The string you want to display
            #             transform=ax.transAxes,  # This makes the position relative to axes (0 to 1 scale)
            #             ha='center',  # Horizontal alignment
            #             va='bottom',  # Vertical alignment
            #             fontsize=12,  # Font size
            #             fontweight='bold'  # Font weight
            #             )

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
                plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)
                # print(res_var)
                print(y_axis)
                ax = function(x='Used_Representation', y=y_axis, data=res_var, size=3)
                ax_config(ax, y_label)
                if problem == "normalized":
                    fig.savefig(f"{final_output_dir}/{name}_{result}_{y_label}.png")
                else:
                    fig.savefig(f"{final_output_dir}/{name}_{datasets_map[problem]}_{y_label}.png")
                plt.close(fig)


if __name__ == '__main__':
    create_plots()
