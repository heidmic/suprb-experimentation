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

            if config["data_directory"] == "mlruns_csv/RBML":
                x_lab = "Estimator"
            if config["data_directory"] == "mlruns_csv/RD":
                x_lab = "RD method"

                new_labels = []
                label_temp = config['heuristics'].values()

                for i, label in enumerate(label_temp):
                    if i % 2 == 0:
                        new_labels.append(label + "\n")  # Nach unten verschieben mit Zeilenumbruch
                    else:
                        new_labels.append("\n" + label)  # Nach oben verschieben

                tick_positions = np.arange(len(label_temp))
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(new_labels)

            if config["data_directory"] == "mlruns_csv/SC" or config["data_directory"] == "mlruns_csv/SAGA":
                x_lab = "SC method"

            if config["data_directory"] == "mlruns_csv/MIX":
                f_index = heuristic.find('f:')
                result = heuristic[f_index+2:]
                result = result.replace('; -e:', '')
                result = result.replace('/', '')
                result = result.replace('CapExperienceWithDimensionality', ' & Experience Cap (dim)')
                result = result.replace('CapExperience', ' & Experience Cap')
                result = result.replace('FilterSubpopulation', '')
                result = result.replace('ExperienceCalculation', '')
                result = result.replace('NBestFitness', r"l Best")
                result = result.replace('NRandom', r"l Random")
                if result == "":
                    result = "Base"
                if result[1] == "&":
                    result = result[2:]

                x_lab = "Number of rules participating"
                ax.set_title(result, style="italic", fontsize=14)

            ax.set_xlabel(x_lab, weight="bold", labelpad=10)

            if y_label != "Complexity":
                y_min = max(0, min(ax.get_yticks()))
                y_max = min(1, max(ax.get_yticks()))
                num_ticks = 7
                if config["data_directory"] == "mlruns_csv/MIX":

                    if result in ["Experience Cap"]:
                        num_ticks = 5
                    elif result == "dummy":
                        num_ticks = 6
                    elif result in ["Experience Cap", "Base", "l Best", "l Best & Experience Cap (dim)"]:
                        num_ticks = 7
                        if result == "Base":
                            y_max = 0.24
                        elif result in ["l Best", "l Best & Experience Cap (dim)"]:
                            y_max = 0.6
                    elif result in ["Experience Cap (dim)", "l Best & Experience Cap"]:
                        num_ticks = 8
                        if result == "l Best & Experience Cap":
                            y_max = 0.7
                    elif result in []:
                        num_ticks = 9
                        if result == "dummy":
                            y_max = 0.8
                    elif result in ["l Random", "l Random & Experience Cap"]:
                        num_ticks = 10
                        if result in ["l Random", "l Random & Experience Cap"]:
                            y_max = 0.9
                    elif result in ["dummy"]:
                        num_ticks = 11
                    elif result in ["RouletteWheel & Experience Cap", "l Best", "RouletteWheel & Experience Cap (dim)", "RouletteWheel", "l Random & Experience Cap (dim)"]:
                        num_ticks = 11
                        if result in ["RouletteWheel & Experience Cap (dim)", "RouletteWheel", "l Random & Experience Cap (dim)"]:
                            y_max = 1
                    else:
                        num_ticks = 7
                elif config["data_directory"] == "mlruns_csv/RBML":
                    if config['datasets'][problem] in ["Combined Cycle Power Plant", "Airfoil Self-Noise", "Concrete Strength"]:
                        num_ticks = 6
                    elif config['datasets'][problem] in ["Energy Efficiency Cooling"]:
                        num_ticks = 6
                elif config["data_directory"] == "mlruns_csv/RD":
                    if config['datasets'][problem] in ["Combined Cycle Power Plant"]:
                        num_ticks = 5
                        if config['datasets'][problem] == "Combined Cycle Power Plant":
                            y_min = 0.055
                            y_max = 0.075
                    elif config['datasets'][problem] in ["Airfoil Self-Noise", "Concrete Strength"]:
                        num_ticks = 6
                        if config['datasets'][problem] == "Airfoil Self-Noise":
                            y_min = 0.1
                    elif config['datasets'][problem] in ["Energy Efficiency Cooling"]:
                        num_ticks = 7
                        y_max = 0.18
                    elif config['datasets'][problem] in ["Parkinson's Telemonitoring"]:
                        num_ticks = 5
                    elif config['datasets'][problem] in ["Physiochemical Properties of Protein Tertiary Structure"]:
                        num_ticks = 7
                        if config['datasets'][problem] == "Physiochemical Properties of Protein Tertiary Structure":
                            y_min = 0.55
                            y_max = 0.85
                elif config["data_directory"] == "mlruns_csv/SAGA":
                    if config['datasets'][problem] in ["Airfoil Self-Noise", "Physiochemical Properties of Protein Tertiary Structure", "Parkinson's Telemonitoring"]:
                        num_ticks = 6
                        if config["datasets"][problem] == "Physiochemical Properties of Protein Tertiary Structure":
                            y_min = 0.6
                    if config['datasets'][problem] in ["Combined Cycle Power Plant"]:
                        num_ticks = 7
                        if config['datasets'][problem] == "Combined Cycle Power Plant":
                            y_min = 0.06
                elif config["data_directory"] == "mlruns_csv/SC":
                    if config['datasets'][problem] in ["Airfoil Self-Noise"]:
                        num_ticks = 6
                        if config['datasets'][problem] == "Airfoil Self-Noise":
                            y_min = 0.05
                            y_max = 0.3
                    elif config['datasets'][problem] in ["Physiochemical Properties of Protein Tertiary Structure"]:
                        num_ticks = 9
                        if config['datasets'][problem] == "Physiochemical Properties of Protein Tertiary Structure":
                            y_min = 0.55
                            y_max = 0.95
                    elif config['datasets'][problem] in ["Combined Cycle Power Plant"]:
                        num_ticks = 5
                        if config['datasets'][problem] == "Combined Cycle Power Plant":
                            y_min = 0.055

                ax.set_ylim(y_min, y_max)
                y_tick_positions = np.linspace(y_min, y_max, num_ticks)
                y_tick_positions = np.round(y_tick_positions, 3)

                print(y_min, y_max, y_tick_positions, config['datasets'][problem])

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
                if config["data_directory"] == "mlruns_csv/MIX":
                    f_index = heuristic.find('f:')
                    result = heuristic[f_index+2:]
                    result = result.replace('; -e:', '')
                    result = result.replace('/', '')
                    result = result.replace('CapExperienceWithDimensionality', ' & Experience Cap (dim)')
                    result = result.replace('CapExperience', ' & Experience Cap')
                    result = result.replace('FilterSubpopulation', '')
                    result = result.replace('ExperienceCalculation', '')
                    result = result.replace('NBestFitness', r"l Best")
                    result = result.replace('NRandom', r"l Random")
                    if result == "":
                        result = "Base"
                    if result[1] == "&":
                        result = result[2:]

                    y_label = "Normalized " + y_label
                    ax.set_xlabel("Number of rules participating", weight="bold", labelpad=10)
                    # ax.set_xlabel(r"l", weight="bold", labelpad=10)
                    ax.set_ylabel(y_label, weight="bold", labelpad=10)
                    fig.savefig(f"{final_output_dir}/{name}_{result}_{y_label}.png")
                else:
                    fig.savefig(f"{final_output_dir}/{name}_{datasets_map[problem]}_{y_label}.png")
                plt.close(fig)

        if config["data_directory"] == "mlruns_csv/MIX":
            return


if __name__ == '__main__':
    create_plots()
