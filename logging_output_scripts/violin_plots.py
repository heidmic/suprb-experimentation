from logging_output_scripts.utils import check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mlflow_utils import get_dataframe

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

heuristics = ['ES', 'RS', 'NS', 'MCNS', 'NSLC']

def create_plots(metricname = 'test_neg_mean_squared_error'):
    metric = 'metrics.' + metricname
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)


dict_list = {}

for problem in datasets:
    res_var = 0
    first = True
    for heuristic in heuristics:
        fold_df = get_dataframe(heuristic, problem)
        if not fold_df.empty:
            name = []
            for x in range(fold_df.shape[0]):
                name.append(heuristic)
            # Adds additional column for plotting
            if first:
                res_var = fold_df.assign(Used_Representation=name)
                first = False
            else:
                res_var = pd.concat([res_var, fold_df.assign(Used_Representation=name)])

            print(f"Done for {problem} with {heuristic}")

    # Invert values since they are stored as negatives
    res_var['metrics.test_neg_mean_squared_error'] = -res_var['metrics.test_neg_mean_squared_error']
    # Store violin-plots of all models in one plot
    fig, ax = plt.subplots()
    ax = sns.violinplot(x='Used_Representation', y='metrics.test_neg_mean_squared_error', data=res_var)
    ax.set(xlabel='RD method')
    ax.set(ylabel='MSE')

    # TODO Change directory
    output_dir = "Violins"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fig.savefig(f"{output_dir}/{problem}.png")
