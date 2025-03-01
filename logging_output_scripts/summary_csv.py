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
    test_r2 = "metrics.test_r2"
    test_accuracy = "metrics.test_accuracy"
    test_f1 = "metrics.test_f1"
    elitist_error ="metrics.elitist_error"
    test_score = "metrics.test_score"
    train_score = "metrics.train_score"
    
    check_and_create_dir(final_output_dir, "csv_summary")
    # Head of csv-File
    if isClass:
        header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP,MEAN_ACC,STD_ACC"
    else:
        header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP,MEAN_R2,STD_R2"
    swapped = False
    if base_model is not None:
        swapped = True
        summary_file_path = f"{final_output_dir}/csv_summary/{config['model_names'][base_model]}_swaps_summary.csv"
        print(f"Creating summary file at: {summary_file_path}")
        with open(summary_file_path, "w") as file:
            file.write(header)
        base_str = f"l:{base_model}"    
        all_runs_list = get_by_config(config, base_str, filter_swapped=False)

    for model, model_name in config['model_names'].items():
        if swapped:
            if model == base_model:
                continue
            model_str = f"n:{model}"
        else:
            model_str = f"l:{model}"
            all_runs_list = get_by_config(config, model_str, filter_swapped=True)
            header += ",MEAN_TEST,STD_TEST"

        fold_df = None
        values = ""
        for problem in config['datasets']:
            values += '\n'
            if swapped:
                values += f"{problem} " + model_str
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
                #if not isClass:
                #    mean_squared_error = -fold_df[mse]
                #    values += "," + str(round(mean_squared_error.mean(), 4))
                #    values += "," + str(round(mean_squared_error.std(), 4))

                if 'test_score' in fold_df:
                    values += "," + str(round(fold_df[test_score].mean(), 4))
                    values += "," + str(round(fold_df[test_score].std(), 4))
                elif 'test_accuracy' in fold_df:
                    values += "," + str(round(fold_df[test_accuracy].mean(), 4))
                    values += "," + str(round(fold_df[test_accuracy].std(), 4))
                elif 'test_r2' in fold_df:
                    values += "," + str(round(fold_df[test_r2].mean(), 4))
                    values += "," + str(round(fold_df[test_r2].std(), 4))
                if not swapped:
                    values += "," + str(round(fold_df[train_score].mean(), 4))
                    values += "," + str(round(fold_df[train_score].std(), 4))

                print(f"Done for {problem} with {model_name}")            
        if not swapped:
            with open(f"{final_output_dir}/csv_summary/{model_name}_summary.csv", "w") as file:
                file.write(header + values)
        else:
            with open(summary_file_path, "a") as file:
                file.write(values)
            

if __name__ == '__main__':
    create_summary_csv(base_model="ridge")
