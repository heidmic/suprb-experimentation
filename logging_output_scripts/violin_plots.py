import json
from logging_output_scripts.utils import check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_output_scripts.utils import get_dataframe, check_and_create_dir


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

#create_output_dir(final_output_dir)


def create_violin_plots():
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)
    check_and_create_dir(config['output_directory'],"violin_plots")
    final_output_dir = f"{config['output_directory']}/violin_plots"
        
    for problem in config['datasets']:
        res_var = 0
        first = True
        for model in config['model_names']:
            fold_df = get_dataframe(all_runs_list=get_all_runs(problem=problem),model=model, dataset=problem)
            if fold_df is not None:
                name = []
                for x in range(fold_df.shape[0]):
                    name.append(model)
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

                print(f"Done for {problem} with {model}")

        # Invert values since they are stored as negatives
        res_var["test_neg_mean_squared_error"] *= -1

        # Store violin-plots of all models in one plot
        fig, ax = plt.subplots()

        ax = sns.violinplot(x='Used_Representation', y="test_neg_mean_squared_error",
                             data=res_var, scale="width", scale_hue=False)
        # ax = sns.swarmplot(x='Used_Representation', y="test_neg_mean_squared_error", data=res_var, size=2)

        ax.set_xlabel('Estimator', weight="bold")
        ax.set_ylabel('MSE', weight="bold")
        title_dict = {"concrete_strength": "Concrete Strength",
                      "combined_cycle_power_plant": "Combined Cycle Power Plant",
                      "airfoil_self_noise": "Airfoil Self Noise",
                      "energy_cool": "Energy Efficiency Cooling"}
        ax.set_title(title_dict[problem], style="italic")

        ax.set_box_aspect(1)

        # fig.savefig(f"{final_output_dir}/{problem}_swarm.png", dpi=500)
        fig.savefig(f"{final_output_dir}/{problem}_violin.png", dpi=500)


if __name__ == '__main__':
    create_violin_plots()
