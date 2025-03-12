import json
from logging_output_scripts.utils import check_and_create_dir, get_all, get_by_config, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logging_output_scripts.utils import get_dataframe, check_and_create_dir

REGRESSOR_CONFIG_PATH = 'logging_output_scripts/config_regression.json'
CLASSIFIER_REGRESSOR_CONFIG_PATH = 'logging_output_scripts/config_classification.json'
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


def create_violin_plots(metric_name="elitist_complexity", isClassifier=False):
    print("STARTING violin-plots")
    config_path = REGRESSOR_CONFIG_PATH
    if isClassifier:
        config_path = CLASSIFIER_REGRESSOR_CONFIG_PATH
    with open(config_path) as f:
        config = json.load(f)
    check_and_create_dir(config['output_directory'],"violin_plots")
    final_output_dir = f"{config['output_directory']}/violin_plots"
    
    #all_runs = get_all()
    for problem in config['datasets']:
        res_var = 0
        first = True
        metric="metrics."+metric_name
        for model in config['model_names']:
            model_name = f"l:{model}"
            fold_df = get_dataframe(all_runs_list=get_by_config(config, problem, filter_swapped=True),
                                     exp_name=model_name, dataset=problem)
            if fold_df is not None:
                #fold_df = fold_df[not fold_df['tags.mlflow.runName'].str.contains("n:")]
                name = []
                for x in range(fold_df.shape[0]):
                    name.append(config['model_names'][model])
                # Adds additional column for plotting
                if first:
                    res_var = fold_df.assign(Used_Representation=name)
                    if metric in res_var.keys():
                        res_var[metric_name] = res_var.pop(metric)
                    first = False
                else:
                    current_res = fold_df.assign(Used_Representation=name)
                    if metric in current_res.keys():
                        current_res[metric_name] = current_res.pop(
                            metric)
                    res_var = pd.concat([res_var, current_res])

                print(f"Done for {problem} with {model}")

        # Invert values since they are stored as negatives
        if metric_name == "test_neg_mean_squared_error":
            res_var["test_neg_mean_squared_error"] *= -1

        # Store violin-plots of all models in one plot
        fig, ax = plt.subplots()

        ax = sns.violinplot(x='Used_Representation', y=metric_name,data=res_var, scale="width", scale_hue=False)
        # ax = sns.swarmplot(x='Used_Representation', y=metric_name, data=res_var, size=2)

        ax.set_xlabel('Estimator', weight="bold")
        
        metric_dict = {"test_neg_mean_squared_error": "MSE",
                       "elitist_complexity": "Complexity",
                       "accuracy": "Accuracy",
                       "test_score": "Score",
                       "training_score": "Accuracy",}       
        ax.set_ylabel(config['metrics'][metric_name], weight="bold")
        #ax.set_ylabel(metric_dict[metric_name], weight="bold")
        
        title_dict = {"concrete_strength": "Concrete Strength",
                      "combined_cycle_power_plant": "Combined Cycle Power Plant",
                      "airfoil_self_noise": "Airfoil Self Noise",
                      "energy_heat": "Energy Efficiency Heating",
                      "breastcancer": "Breast Cancer",
                      "raisin": "Raisin",
                      "abalone": "Abalone"}
        ax.set_title(title_dict[problem], style="italic")
        ax.set_box_aspect(1)
        fig.savefig(f"{final_output_dir}/{problem}_swarm.png", dpi=500)
        #fig.savefig(f"{final_output_dir}/{problem}_{metric_name}.png", dpi=500)


if __name__ == '__main__':
    create_violin_plots(metric_name="elitist_complexity")
