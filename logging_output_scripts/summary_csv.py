from logging_output_scripts.utils import get_dataframe, create_output_dir, config


"""
Extracts values from csv-Files gained from mlflow, performs calculations and
stores the results in a new csv-File for all Models specified
Leave out/add metrics that you want to evaluate
"""

final_output_dir = f"{config['output_directory']}/csv_summary"
create_output_dir(config['output_directory'])
create_output_dir(final_output_dir)


def create_summary_csv():
    for heuristic in config['heuristics']:
        # Head of csv-File
        header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP," \
            f"MEAN_MSE,STD_MSE"

        values = "\n"
        for problem in config['datasets']:
            values += problem

            print(f"WORKING ON DATASET {problem} WITH {heuristic}")
            fold_df = get_dataframe(heuristic, problem)

            if "metrics.elitist_complexity" in fold_df.keys():
                elitist_complexity = "metrics.elitist_complexity"
            else:
                elitist_complexity = "elitist_complexity"

            if "metrics.test_neg_mean_squared_error" in fold_df.keys():
                test_neg_mean_squared_error = "metrics.test_neg_mean_squared_error"
            else:
                test_neg_mean_squared_error = "test_neg_mean_squared_error"

            # Calculates mean, min, max, median and std of elitist_complexity across all runs
            # elitist_complexity_eval = fold_df[elitist_complexity]
            # values += "," + str(elitist_complexity_eval.min())
            # values += "," + str(elitist_complexity_eval.max())
            # values += "," + str(round(elitist_complexity_eval.mean(), 2))
            # values += "," + str(round(elitist_complexity_eval.std(), 2))
            # values += "," + str(elitist_complexity_eval.median())

            # Calculates both mse and std of mse between all runs (Changes MSE to positive value)
            mse = -fold_df[test_neg_mean_squared_error]
            values += "," + str(round(mse.mean(), 4))
            values += "," + str(round(mse.std(), 4))

            values += '\n\n'

        print(f"{heuristic} FINISHED")
        with open(f"{final_output_dir}/{heuristic}_summary.csv", "w") as file:
            file.write(header + values)


if __name__ == '__main__':
    create_summary_csv()
