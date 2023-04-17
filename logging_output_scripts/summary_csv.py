from logging_output_scripts.utils import get_dataframe, create_output_dir, config
import numpy as np

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

            # if problem == "combined_cycle_power_plant":
            #     fold_df[elitist_complexity] = np.array(
            #         [551, 575, 541, 545, 559, 551, 563, 574, 542, 567, 555, 572, 542, 555, 553, 521, 563, 567, 565, 536,
            #          584, 622, 587, 586, 537, 554, 512, 551, 557, 545, 544, 547, 589, 569, 563, 539, 557, 517, 547, 546,
            #          556, 552, 538, 552, 558, 533, 573, 546, 563, 555, 550, 559, 548, 549, 534, 565, 561, 571, 570, 570,
            #          577, 589, 554, 539])
            # elif problem == "airfoil_self_noise":
            #     fold_df[elitist_complexity] = np.array(
            #         [68, 67, 61, 66, 67, 68, 68, 71, 64, 69, 66, 66, 72, 65, 70, 66, 71, 67, 69, 71, 68, 67, 67, 70, 71,
            #          68, 65, 63, 70, 68, 68, 67, 66, 67, 71, 66, 66, 69, 72, 65, 69, 65, 75, 70, 65, 70, 67, 67, 71, 64,
            #          67, 70, 72, 72, 68, 68, 67, 72, 65, 68, 66, 66, 70, 67])
            # elif problem == "concrete_strength":
            #     fold_df[elitist_complexity] = np.array(
            #         [108, 115, 106, 103, 114, 102, 117, 120, 108, 115, 106, 103, 114, 102, 117, 120, 108, 115, 106, 103,
            #          114, 102, 117, 120, 108, 115, 106, 103, 114, 102, 117, 120, 108, 115, 106, 103, 114, 102, 117, 120,
            #          108, 115, 106, 103, 114, 102, 117, 120, 108, 115, 106, 103, 114, 102, 117, 120, 108, 115, 106, 103,
            #          114, 102, 117, 120])
            # elif problem == "energy_cool":
            #     fold_df[elitist_complexity] = np.array(
            #         [26, 27, 27, 27, 27, 27, 25, 26, 26, 27, 27, 27, 27, 27, 25, 26, 26, 27, 27, 27, 27, 27, 25, 26, 26,
            #          27, 27, 27, 27, 27, 25, 26, 26, 27, 27, 27, 27, 27, 25, 26, 26, 27, 27, 27, 27, 27, 25, 26, 26, 27,
            #          27, 27, 27, 27, 25, 26, 26, 27, 27, 27, 27, 27, 25, 26])

            # Calculates mean, min, max, median and std of elitist_complexity across all runs
            elitist_complexity_eval = fold_df[elitist_complexity]
            values += "," + str(elitist_complexity_eval.min())
            values += "," + str(elitist_complexity_eval.max())
            values += "," + str(round(elitist_complexity_eval.mean(), 2))
            values += "," + str(round(elitist_complexity_eval.std(), 2))
            values += "," + str(elitist_complexity_eval.median())

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
