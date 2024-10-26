from logging_output_scripts.utils import get_csv_df, get_normalized_df, check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from utils import datasets_map
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
plt.rcParams['figure.dpi'] = 200

plt.tight_layout()


mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"


def create_plots():
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"

    # for output_dir in ["violin_plots", "swarm_plots", "line_plots"]:
    #     check_and_create_dir(final_output_dir, output_dir)

    final_output_dir = f"{config['output_directory']}"
    scaler = MinMaxScaler()

    for problem in config['datasets']:
        # for yy in [0]:
        first = True
        res_var = 0
        counter = 0
        fold_df = None

        for heuristic, renamed_heuristic in config['heuristics'].items():
            # fold_df = get_normalized_df(heuristic)
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
            x_lab = "Number of rules participating" if config["normalize_datasets"] else "Estimator"
            # x_lab = r"l"
            ax.set_xlabel(x_lab, weight="bold", fontstyle='italic')
            ax.set_ylabel(y_label, weight="bold")
            ax.set_title(config['datasets'][problem] if not config["normalize_datasets"] else result, style="italic")
            # ax.set_title(result, style="italic")
            # ax.set_box_aspect(1)

        # problem = problem if not config["normalize_datasets"] else "normalized"
        # problem = "normalized"

        ################### MSE ###########################
        plots = {  # "violin": sns.violinplot,
            "swarm": sns.swarmplot,
            #  "box": sns.boxplot
        }

        y_axis_label = {"MSE": mse,
                        "Complexity": complexity
                        }

        # y_axis_label = {"Normalized MSE": mse,
        #                 "Normalized Complexity": complexity
        #                 }

        f_index = heuristic.find('f:')
        result = heuristic[f_index+2:]
        # result = result.replace('; -e:', '')
        # result = result.replace('/', '')
        # result = result.replace('CapExperienceWithDimensionality', ' & Experience Cap (dim)')
        # result = result.replace('CapExperience', ' & Experience Cap')
        # result = result.replace('FilterSubpopulation', '')
        # result = result.replace('ExperienceCalculation', '')
        # result = result.replace('NBestFitness', r"l Best")
        # result = result.replace('NRandom', r"l Random")
        # if result == "":
        #     result = "Base"
        # if result[1] == "&":
        #     result = result[2:]

        for name, function in plots.items():
            for y_label, y_axis in y_axis_label.items():
                fig, ax = plt.subplots(dpi=400)
                ax = function(x='Used_Representation', y=y_axis, data=res_var, size=3)
                ax_config(ax, y_label)
                if problem == "normalized":
                    fig.savefig(f"{final_output_dir}/{name}_{result}_{y_label}.png")
                else:
                    fig.savefig(f"{final_output_dir}/{name}_{datasets_map[problem]}_{y_label}.png")
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
