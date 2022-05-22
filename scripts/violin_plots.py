import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""Uses seaborn-package to create violin-Plots comparing model performances
on multiple datasets
"""

dirs = {0 : 'OBR', 1 : 'UBR', 2 : 'CSR', 3 : 'MPR'}

datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}

dict_list = {}

for i in datasets:
    problem = datasets[i]
    res_var = 0
    for j in dirs:
        current_dir = dirs[j]
        df = pd.read_csv(f"../{current_dir}/{problem}.csv")
        fold_df = df[df['Name'].str.contains('fold')]
        name = []
        for x in range (fold_df.shape[0]):
            name.append(current_dir)
        if j == 0:
            res_var = fold_df.assign(Used_Representation=name)
        else:
            res_var = pd.concat([res_var, fold_df.assign(Used_Representation=name)])

    res_var['test_neg_mean_squared_error'] = -res_var['test_neg_mean_squared_error']
    # Store violin-plots of all models in one plot
    fig, ax = plt.subplots()
    ax = sns.violinplot(x='Used_Representation', y='test_neg_mean_squared_error', palette="muted", data=res_var)
    ax.set(ylabel='MSE')
    fig.savefig(f'Violins/{problem}.png')
