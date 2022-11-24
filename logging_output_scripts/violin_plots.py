import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_output_scripts.utils import get_dataframe, create_output_dir, config


"""
Uses seaborn-package to create violin-Plots comparing model performances
on multiple datasets
"""
sns.set_style("whitegrid")

final_output_dir = f"{config['output_directory']}/violin_plots"
create_output_dir(config['output_directory'])
create_output_dir(final_output_dir)

if config['filetype'] == 'csv':
    test_neg_mean_squared_error = "test_neg_mean_squared_error"
elif config['filetype'] == 'mlflow':
    test_neg_mean_squared_error = "metrics.test_neg_mean_squared_error"


def create_violin_plots():
    for problem in config['datasets']:
        res_var = 0
        first = True
        for heuristic in config['heuristics']:
            fold_df = get_dataframe(heuristic, problem)
            if fold_df is not None:
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
        res_var[test_neg_mean_squared_error] = -res_var[test_neg_mean_squared_error]
        # Store violin-plots of all models in one plot
        fig, ax = plt.subplots()
        ax = sns.violinplot(x='Used_Representation', y=test_neg_mean_squared_error, data=res_var)
        ax.set(xlabel='RD method')
        ax.set(ylabel='MSE')

        fig.savefig(f"{final_output_dir}/{problem}.png")


if __name__ == '__main__':
    create_violin_plots()
