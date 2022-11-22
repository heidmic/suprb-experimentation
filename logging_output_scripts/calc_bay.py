import os
import numpy as np
from baycomp import SignedRankTest
from itertools import combinations
from mlflow_utils import get_dataframe


"""
Uses baycomp-package to calculate probabilities of one model performing
better than another (based on MSE). Comparisons are performed on all
possible combinations for all models
NOTE: If you need to compare more than two models at once please refer to: https://github.com/dpaetzel/cmpbayes
(Using a nix development environment is recommended for cmpbayes)
"""

heuristics = ['ES', 'RS', 'NS', 'MCNS', 'NSLC']

datasets = ["concrete_strength", 'combined_cycle_power_plant',
            'airfoil_self_noise', 'energy_cool']

if config['filetype'] == 'csv':
    test_neg_mean_squared_error = "test_neg_mean_squared_error"
elif config['filetype'] == 'mlflow':
    test_neg_mean_squared_error = "metrics.test_neg_mean_squared_error"

# Returns dictionary where keys respond to the used model and items are array of MSE for all problems


def load_error_list():
    dict_list = {}

    for heuristic in heuristics:
        res_var = []
        for problem in datasets:
            fold_df = get_dataframe(heuristic, problem)
            if not fold_df.empty:
                mse_df = -fold_df['metrics.test_neg_mean_squared_error'].mean()
                res_var.append(mse_df)
        dict_list.update({heuristic: np.array(res_var)})
    return dict_list


# Performs calculation, stores both simplex plots and csv File
def calc_bayes(save_csv: bool = True, save_plots: bool = True, rope: float = 0.1):
    header = "COMPARED REPS, PLEFT, ROPE, PRIGHT\n"
    error_list = load_error_list()
    # Matches each possible combination
    result_list = list(map(dict, combinations(error_list.items(), 2)))

    # Compare all unique combinations between the used models
    for combination in result_list:
        dir_1 = list(combination.keys())[0]
        dir_2 = list(combination.keys())[1]
        comparison_title = dir_1 + " VS. " + dir_2
        # Perform SignedRankTest on average MSE for two Models
        posterior = SignedRankTest(x=error_list[dir_1], y=(error_list[dir_2]), rope=rope)
        print(f"Considered values - {dir_1} : {error_list[dir_1]} vs {dir_2} : {error_list[dir_2]} ")
        probabilities = posterior.probs()
        print(f"Probabilities: {probabilities[0]} {probabilities[1]} {probabilities[2]}")
        # Plot simplex, histogram doesnt work properly
        simplex_plot = posterior.plot_simplex(names=(dir_1, "rope", dir_2))
        if save_plots:
            simplex_plot.savefig(f'{final_output_dir}/{comparison_title}.png')

        # Store probabilities for each comparison as a row
        header += f"{comparison_title}, {probabilities[0]}, {probabilities[1]}, {probabilities[2]}\n"

    if save_csv:
        with open(f"{final_output_dir}/Resulting_probs.csv", "w") as file:
            file.write(header)


if __name__ == '__main__':
    calc_bayes()
