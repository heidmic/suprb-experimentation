import pandas as pd
import numpy as np
from baycomp import SignedRankTest
import matplotlib.pyplot as plt

"""
Uses baycomp-package to calculate probabilities of one model performing
better than another (based on MSE). Comparisons are performed on all
possible combinations for all models
"""

# Header
s = "COMPARED REPS, PLEFT, ROPE, PRIGHT"

# Used Models
dirs = {0 : 'OBR', 1 : 'UBR', 2 : 'CSR', 3 : 'MPR'}

# Used Datasets
datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}

dict_list = {}

for i in dirs:
    current_dir = dirs[i]
    res_var = np.zeros((len(datasets)))
    for j in datasets:
        problem = datasets[j]
        df = pd.read_csv(f"../{current_dir}/{problem}.csv")
        fold_df = df[df['Name'].str.contains('fold')]
        mse = -fold_df['test_neg_mean_squared_error'].mean()
        res_var[j] = mse
    a = {current_dir : res_var}
    dict_list.update(a)


result_dict = {}

for i in dirs:
    s += "\n"
    for j in dirs:
        dir_1 = dirs[i]
        dir_2 = dirs[j]
        # Avoid comparing identical models
        if dir_1 == dir_2:
            continue
        else:
            temp_string = dir_1 + " VS. " + dir_2
            # Perform SignedRankTest on average MSE for two Models
            posterior = SignedRankTest(x=dict_list[dir_1], y=(dict_list[dir_2]), rope=0.1)
            print(f"Considered values - {dir_1} : {dict_list[dir_1]} vs {dir_2} : {dict_list[dir_2]} ")
            temp_vals = posterior.probs()
            print(f"Probs: {temp_vals[0]} {temp_vals[1]} {temp_vals[2]}")
            # Plot simplex, histogram doesnt work properly
            res_plot = posterior.plot_simplex(names=(dir_1, "rope", dir_2))
            result_dict[temp_string] = temp_vals
            # Store probabilities for each comparison as a row
            s += f"{temp_string}, {temp_vals[0]}, {temp_vals[1]}, {temp_vals[2]}\n"
            res_plot.savefig(f'Bayesian/{temp_string}.png')


file = open("Bayesian/Resulting_probs.csv", "w")


file.write(s)
