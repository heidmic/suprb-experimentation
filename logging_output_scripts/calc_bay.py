import pandas as pd
import numpy as np
from baycomp import SignedRankTest
from itertools import combinations

"""
Uses baycomp-package to calculate probabilities of one model performing
better than another (based on MSE). Comparisons are performed on all
possible combinations for all models
NOTE: If you need to compare more than two models at once please refer to: https://github.com/dpaetzel/cmpbayes
(Using a nix development environment is recommended for cmpbayes)
"""

# Used Models (Adjust to reflect folders)
dirs = ['OBR', 'UBR', 'CSR', 'MPR']

# Used Datasets (Refers to .csv - File stored in folder)
datasets = ['parkinson_total', 'protein_structure', 'airfoil_self_noise',
            'concrete_strength', 'combined_cycle_power_plant']


# Returns dictionary where keys respond to the used model and items are array of MSE for all problems
def load_error_list():
    dict_list = {}

    for current_dir in dirs:
        res_var = []
        for problem in datasets:
            # Folder in same directory as logging_output_scripts-Folder
            df = pd.read_csv(f"../{current_dir}/{problem}.csv")
            fold_df = df[df['Name'].str.contains('fold')]
            mse_df = -fold_df['test_neg_mean_squared_error'].mean()
            res_var.append(mse_df)
        dict_list.update({current_dir: res_var})
    return dict_list


# Performs calculation, stores both simplex plots and csv File
def calc_bayes(save_csv: bool = True, save_plots: bool = True, rope: float = 0.1):
    header = "COMPARED REPS, PLEFT, ROPE, PRIGHT\n"
    dict_list = load_error_list()
    # Matches each possible combination
    result_list = list(map(dict, combinations(dict_list.items(), 2)))

    # Compare all unique combinations between the used models
    for combination in result_list:
        dir_1 = list(combination.keys())[0]
        dir_2 = list(combination.keys())[1]
        comparison_title = dir_1 + " VS. " + dir_2
        # Perform SignedRankTest on average MSE for two Models
        posterior = SignedRankTest(x=dict_list[dir_1], y=(dict_list[dir_2]), rope=rope)
        print(f"Considered values - {dir_1} : {dict_list[dir_1]} vs {dir_2} : {dict_list[dir_2]} ")
        probabilities = posterior.probs()
        print(f"Probs: {probabilities[0]} {probabilities[1]} {probabilities[2]}")
        # Plot simplex, histogram doesnt work properly
        simplex_plot = posterior.plot_simplex(names=(dir_1, "rope", dir_2))
        if save_plots:
            simplex_plot.savefig(f'Bayesian/{comparison_title}.png')

        # Store probabilities for each comparison as a row
        header += f"{comparison_title}, {probabilities[0]}, {probabilities[1]}, {probabilities[2]}\n"

    if save_csv:
        with open("Bayesian/Resulting_probs.csv", "w") as file:
            file.write(header)


if __name__ == '__main__':
    calc_bayes()
