from logging_output_scripts.utils import get_dataframe, check_and_create_dir, get_all_runs
import json
"""
Extracts values from csv-Files gained from mlflow, performs calculations and
stores the results in a new csv-File for all Models specified
Leave out/add metrics that you want to evaluate
"""
with open('config.json') as f:
    config = json.load(f)
final_output_dir = f"{config['output_directory']}"
elitist_complexity = "metrics.elitist_complexity"
mse = "metrics.population_error_mean"


def create_summary_csv():
    all_runs_list = get_all_runs("")
    for heuristic, renamed_heuristic in config['heuristics'].items():
        # Head of csv-File
        header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP,MEAN_MSE,STD_MSE"

        values = "\n"
        for problem in config['datasets']:
            values += problem
            fold_df = get_dataframe(all_runs_list, heuristic, problem)

            if fold_df is not None:
                # Calculates mean, min, max, median and std of elitist_complexity across all runs
                elitist_complexity_eval = fold_df[elitist_complexity]
                values += "," + str(elitist_complexity_eval.min())
                values += "," + str(elitist_complexity_eval.max())
                values += "," + str(round(elitist_complexity_eval.mean(), 2))
                values += "," + str(round(elitist_complexity_eval.std(), 2))
                values += "," + str(elitist_complexity_eval.median())

                # Calculates both mse and std of mse between all runs (Changes MSE to positive value)
                mean_squared_error = -fold_df[mse]
                values += "," + str(round(mean_squared_error.mean(), 4))
                values += "," + str(round(mean_squared_error.std(), 4))

                values += '\n\n'

                print(f"Done for {problem} with {renamed_heuristic}")

        with open(f"{final_output_dir}/csv_summary/{renamed_heuristic}_summary.csv", "w") as file:
            file.write(header + values)


if __name__ == '__main__':
    check_and_create_dir(final_output_dir, "csv_summary")

    create_summary_csv()