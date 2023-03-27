from logging_output_scripts.utils import check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_output_scripts.utils import get_dataframe, create_output_dir, config


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

final_output_dir = f"{config['output_directory']}/violin_plots"
create_output_dir(config['output_directory'])
create_output_dir(final_output_dir)


def create_violin_plots():
    for problem in config['datasets']:
        res_var = 0
        first = True
        for heuristic in config['heuristics']:
            fold_df = get_dataframe(heuristic, problem)
            if fold_df is not None:
                name = []
                for x in range(fold_df.shape[0]):
                    if heuristic == "NS" or heuristic == "MCNS" or heuristic == "NSLC":
                        heuristic += "-G"
                    name.append(heuristic)
                # Adds additional column for plotting
                if first:
                    res_var = fold_df.assign(Used_Representation=name)
                    if "metrics.test_neg_mean_squared_error" in res_var.keys():
                        res_var["test_neg_mean_squared_error"] = res_var.pop("metrics.test_neg_mean_squared_error")
                    first = False
                else:
                    current_res = fold_df.assign(Used_Representation=name)
                    if "metrics.test_neg_mean_squared_error" in current_res.keys():
                        current_res["test_neg_mean_squared_error"] = current_res.pop(
                            "metrics.test_neg_mean_squared_error")
                    res_var = pd.concat([res_var, current_res])

                print(f"Done for {problem} with {heuristic}")

        # Invert values since they are stored as negatives
        res_var["test_neg_mean_squared_error"] *= -1

        # Store violin-plots of all models in one plot
        fig, ax = plt.subplots()
        ax = sns.violinplot(x='Used_Representation', y="test_neg_mean_squared_error",
                            data=res_var, scale="width", scale_hue=False)
        ax.set(xlabel='Estimator')
        ax.set(ylabel='MSE')
        ax.set(title=problem)

        plt.tight_layout()
        fig.savefig(f"{final_output_dir}/{problem}.png")


if __name__ == '__main__':
    create_violin_plots()
