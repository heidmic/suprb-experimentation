import json
from logging_output_scripts.utils import get_by_config, get_dataframe, check_and_create_dir, get_df

"""
Extracts values from csv-Files gained from mlflow, performs calculations and
stores the results in a new csv-File for all Models specified
Leave out/add metrics that you want to evaluate
"""

CONFIG_PATH = 'logging_output_scripts/config.json'
CLASS_CONFIG_PATH = 'logging_output_scripts/config_class.json'


def create_summary_csv(isClass=False, base_model = None):
    config_path = CONFIG_PATH
    if isClass:
        config_path = CLASS_CONFIG_PATH
    with open(config_path) as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"
    elitist_complexity = "metrics.elitist_complexity"
    mse = "metrics.test_neg_mean_squared_error"
    elitist_error ="metrics.elitist_error"
    
    swapped = False
    if base_model is not None:
        swapped = True
        all_runs_list = get_by_config(config, base_model, filter_swapped=False)
        
    check_and_create_dir(final_output_dir, "csv_summary")
    for model, model_name in config['model_names'].items():
        # Head of csv-File
        header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP,MEAN_MSE,STD_MSE"
        header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP,MEAN_ERROR,STD_ERROR"

        if swapped:
            model_str = f"n:{model}"
        else:
            model_str = f"l:{model}"
            all_runs_list = get_by_config(config, model_str, filter_swapped=True)

        fold_df = None
        values = "\n"
        for problem in config['datasets']:
            if swapped:
                values += f"{problem}" + model_str
            else:
                values += problem
            #fold_df = get_df(model, problem)
            fold_df = get_dataframe(all_runs_list, model_str, problem)

            if fold_df is not None:
                # Calculates mean, min, max, median and std of elitist_complexity across all runs
                elitist_complexity_eval = fold_df[elitist_complexity]
                values += "," + str(elitist_complexity_eval.min())
                values += "," + str(elitist_complexity_eval.max())
                values += "," + str(round(elitist_complexity_eval.mean(), 2))
                values += "," + str(round(elitist_complexity_eval.std(), 2))
                values += "," + str(elitist_complexity_eval.median())

                # Calculates both mse and std of mse between all runs (Changes MSE to positive value)
                #mean_squared_error = -fold_df[mse]
                #values += "," + str(round(mean_squared_error.mean(), 4))
                #values += "," + str(round(mean_squared_error.std(), 4))
                
                error = fold_df[elitist_error]
                values += "," + str(round(error.mean(), 4))
                values += "," + str(round(error.std(), 4))

                values += '\n'

                print(f"Done for {problem} with {model_name}")
        if not swapped:
            with open(f"{final_output_dir}/csv_summary/{model_name}_summary.csv", "w") as file:
                file.write(header + values)
    if swapped:
        with open(f"{final_output_dir}/csv_summary/{base_model}_swaps_summary.csv", "w") as file:
            file.write(header + values)

if __name__ == '__main__':
    create_summary_csv(base_model="l1")
