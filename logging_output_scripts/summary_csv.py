from mlflow_utils import get_dataframe


"""
Extracts values from csv-Files gained from mlflow, performs calculations and
stores the results in a new csv-File for all Models specified
Leave out/add metrics that you want to evaluate
"""

elitist_complexity = "metrics.elitist_complexity"
mse = "metrics.test_neg_mean_squared_error"


def create_summary_csv():
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)
    final_output_dir = f"{config['output_directory']}"
    check_and_create_dir(final_output_dir, "csv_summary")

        print(f"WORKING ON DATASET {problem} WITH {directory}")
        fold_df = get_dataframe(directory, problem)

        # Calculates mean, min, max, median and std of elitist_complexity across all runs
        elitist_complexity = fold_df['metrics.elitist_complexity']
        values += "," + str(elitist_complexity.min())
        values += "," + str(elitist_complexity.max())
        values += "," + str(round(elitist_complexity.mean(), 2))
        values += "," + str(round(elitist_complexity.std(), 2))
        values += "," + str(elitist_complexity.median())

        # Calculates both mse and std of mse between all runs (Changes MSE to positive value)
        mse = -fold_df['metrics.test_neg_mean_squared_error']
        values += "," + str(round(mse.mean(), 4))
        values += "," + str(round(mse.std(), 4))

                print(f"Done for {problem} with {renamed_heuristic}")

    # TODO Change directory
    print(f"{directory} FINISHED")
    with open(f"csv_summary/{directory}_summary.csv", "w") as file:
        file.write(header + values)
