import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
Uses seaborn-package to create violin-Plots comparing model performances
on multiple datasets
"""

dirs = {0: 'OBR', 1: 'UBR', 2: 'CSR', 3: 'MPR'}

datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}

dict_list = {}

for problem in datasets.values():
    res_var = 0
    first = True
    for directory in dirs.values():
        df = pd.read_csv(f"../{directory}/{problem}.csv")
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
    ax = sns.violinplot(x='Used_Representation', y='test_neg_mean_squared_error', palette="muted", data=res_var)
    ax.set(ylabel='MSE')

    # Create folder if not yet done
    directory = "Violins"
    if not os.path.exists(directory):
        os.mkdir(directory)

    fig.savefig(f'Violins/{problem}.png')
