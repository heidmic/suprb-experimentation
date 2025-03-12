import json
from logging_output_scripts.utils import get_all_dataframe, get_by_config, get_dataframe, check_and_create_dir, get_df
import pandas as pd

"""
Extracts values from csv-Files gained from mlflow, performs calculations and
stores the results in a new csv-File for all Models specified
Leave out/add metrics that you want to evaluate
"""

REGRESSOR_CONFIG_PATH = 'logging_output_scripts/config_regression.json'
CLASSIFIER_REGRESSOR_CONFIG_PATH = 'logging_output_scripts/config_classification.json'


def create_summary_csv(isClassifier=False, base_model = None):
    print("STARTING summary csv")
    config_path = REGRESSOR_CONFIG_PATH
    if isClassifier:
        config_path = CLASSIFIER_REGRESSOR_CONFIG_PATH
    with open(config_path) as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"
    
    check_and_create_dir(final_output_dir, "csv_summary")
    # Head of csv-File
    header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP"
    swapped = False
    if base_model is not None:
        swapped = True
        if isClassifier:
            header += ",MEAN_ACC,STD_ACC"
        else:
            header += ",MEAN_R2,STD_R2"
        summary_file_path = f"{final_output_dir}/csv_summary/{config['model_names'][base_model]}_swaps_summary.csv"
        print(f"Creating summary file at: {summary_file_path}")
        with open(summary_file_path, "w") as file:
            file.write(header)
        base_str = f"l:{base_model}"    
        all_runs_list = get_by_config(config, base_str, filter_swapped=False)
    else:
        if isClassifier:
            header += ",MEAN_ACC,STD_ACC,MEAN_F1,STD_F1"
        else:
            header += ",MEAN_R2,STD_R2,MEAN_MSE,STD_MSE"
        header += ",MEAN_TRAIN,STD_TRAIN"

    for model, model_name in config['model_names'].items():
        if swapped:
            if model == base_model:
                continue
            model_str = f"n:{model}"
        else:
            model_str = f"l:{model}"
            all_runs_list = get_by_config(config, model_str, filter_swapped=True)

        values = ""
        for problem in config['datasets']:
            values += log_values(all_runs_list, model_str, problem, log_comp=True, swapped=swapped)
            print(f"Done for {problem} with {model_name}") 
                    
        if not swapped:
            with open(f"{final_output_dir}/csv_summary/{model_name}_summary.csv", "w") as file:
                file.write(header + values)
        else:
            with open(summary_file_path, "a") as file:
                file.write(values)

def log_values(all_runs_list, model_str, problem, log_comp = True, swapped = False):
    elitist_complexity = "metrics.elitist_complexity"
    mse = "metrics.test_neg_mean_squared_error"
    test_r2 = "metrics.test_r2"
    test_accuracy = "metrics.test_accuracy"
    test_f1 = "metrics.test_f1"
    elitist_error ="metrics.elitist_error"
    test_score = "metrics.test_score"
    training_score = "metrics.training_score"
    val = '\n'
    if swapped:
        val += f"{problem} " + model_str
    else:
        val += problem
    if swapped:
        fold_df = get_all_dataframe(all_runs_list, model_str, problem)
    else:
        fold_df = get_dataframe(all_runs_list, model_str, problem)
    if fold_df is not None:
        #print(f"Model: {model_str}")
        #print(f"Shape of fold_df: {fold_df.shape}")
        # Calculates mean, min, max, median and std of elitist_complexity across all runs
        if log_comp:
            elitist_complexity_eval = fold_df[elitist_complexity]
            val += "," + str(elitist_complexity_eval.min())
            val += "," + str(elitist_complexity_eval.max())
            val += "," + str(round(elitist_complexity_eval.mean(), 2))
            val += "," + str(round(elitist_complexity_eval.std(), 2))
            val += "," + str(elitist_complexity_eval.median())

        if swapped:
            #print("Scores: " + fold_df[test_score].to_string())
            #values += "," + str(round(fold_df[elitist_error].mean(), 4))
            #values += "," + str(round(fold_df[elitist_error].std(), 4))
            val += "," + str(round(fold_df[test_score].mean(), 4))
            val += "," + str(round(fold_df[test_score].std(), 4))
        else:
            # Single Column
            if test_score in fold_df:
                print("Single test score")
                val += "," + str(round(fold_df[test_score].mean(), 4))
                val += "," + str(round(fold_df[test_score].std(), 4))
            # First column
            if test_accuracy in fold_df:
                print("accuracy recorded: " + str(round(fold_df[test_accuracy].mean(), 4)))
                if str(fold_df[test_accuracy].mean()) == "nan":
                    val += ",-1,-1"
                else:
                    val += "," + str(round(fold_df[test_accuracy].mean(), 4))
                    val += "," + str(round(fold_df[test_accuracy].std(), 4))
            if test_r2 in fold_df:
                val += "," + str(round(fold_df[test_r2].mean(), 4))
                val += "," + str(round(fold_df[test_r2].std(), 4))
            # Second column
            if test_f1 in fold_df:
                print("f1 recorded: " + str(round(fold_df[test_f1].mean(), 4)))
                if str(fold_df[test_f1].mean()) == "nan":
                    val += ",-1,-1"
                else:
                    val += "," + str(round(fold_df[test_f1].mean(), 4))
                    val += "," + str(round(fold_df[test_f1].std(), 4))
            if mse in fold_df:
                val += "," + str(round(-fold_df[mse].mean(), 4))
                val += "," + str(round(-fold_df[mse].std(), 4))
            # Last column
            if not swapped:
                val += "," + str(round(fold_df[training_score].mean(), 4))
                val += "," + str(round(fold_df[training_score].std(), 4))
    return val


def log_alternate(isClassifier = True):
    config_path = REGRESSOR_CONFIG_PATH
    if isClassifier:
        config_path = CLASSIFIER_REGRESSOR_CONFIG_PATH
    with open(config_path) as f:
        config = json.load(f)
    model = "Tree"
    conf = {
        "data_directory": "mlruns",
        "model_names":  {
            model: model
        }
    }
    list=get_by_config(conf, model)
    val = ""
    for problem in config['datasets']:
        val += log_values(list, model, problem, log_comp=False, swapped=True)
        print(val)
        header = "Problem, Score, Std_Score"
        final_output_dir = f"{config['output_directory']}"
    with open(f"{final_output_dir}/csv_summary/Forest_summary.csv", "w") as file:
            file.write(header + val)
                               

if __name__ == '__main__':
    #log_alternate()
    create_summary_csv(isClassifier=True)
