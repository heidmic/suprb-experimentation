import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
Uses seaborn-package to create violin-Plots comparing model performances
on multiple datasets
"""
sns.set_style("whitegrid")

path_to_csvs = r"C:\Users\m\Documents\SupRB\rule_discovery_paper\run_csvs"
plots = r"C:\Users\m\Documents\SupRB\rule_discovery_paper"

heur = ['ES', 'RS', 'NS', 'MCNS', 'NSLC']

datasets = ["concrete_strength", 'combined_cycle_power_plant',
            'airfoil_self_noise', 'energy_cool']

dict_list = {}

for problem in datasets:
    res_var = 0
    first = True
    for directory in heur:
        df = pd.read_csv(f"{path_to_csvs}/{directory}/{problem}.csv")
        fold_df = df[df['Name'].str.contains('fold')]
        name = []
        for x in range(fold_df.shape[0]):
            name.append(directory)
        # Adds additional column for plotting
        if first:
            res_var = fold_df.assign(Used_Representation=name)
            first = False
        else:
            res_var = pd.concat([res_var, fold_df.assign(Used_Representation=name)])

    # Invert values since they are stored as negatives
    res_var['test_neg_mean_squared_error'] = -res_var['test_neg_mean_squared_error']
    # Store violin-plots of all models in one plot
    fig, ax = plt.subplots()
    ax = sns.violinplot(x='Used_Representation', y='test_neg_mean_squared_error', data=res_var)
    ax.set(xlabel='RD method')
    ax.set(ylabel='MSE')

    # Create folder if not yet done
    directory = "Violins"
    if not os.path.exists(plots+'\\'+directory):
        os.mkdir(plots+'\\'+directory)

    fig.savefig(fr'{plots}/Violins/{problem}.png')
