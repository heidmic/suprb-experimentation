import pandas as pd

"""
Extracts values from csv-Files gained from mlflow, performs calculations and
stores the results in a new csv-File for all Models specified
Leave out/add metrics that you want to evaluate
"""

# Datasets runs were performed on, responds to one csv file each
datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}
# The used representations (or models) on which runs were performed
dirs = {0: 'OBR', 1: 'UBR', 2: 'CSR', 3: 'MPR'}

for directory in dirs.values():
    # Head of csv-File
    header = f"Problem,MIN_COMP,MAX_COMP,MEAN_COMP,STD_COMP,MEDIAN_COMP," \
             f"MEAN_MSE,STD_MSE"

    values = "\n"
    for problem in datasets.values():
        values += problem

        print(f"WORKING ON DATASET {problem} WITH {directory}")
        # Read from csv-File in directory named after model and named after dataset
        df = pd.read_csv(f"../{directory}/{problem}.csv")
        # Filter out individual runs (Removes averaged values)
        fold_df = df[df['Name'].str.contains('fold')]

        # Calculates mean, min, max, median and std of elitist_complexity across all runs
        elitist_complexity = fold_df['elitist_complexity']
        values += "," + str(elitist_complexity.min())
        values += "," + str(elitist_complexity.max())
        values += "," + str(round(elitist_complexity.mean(), 2))
        values += "," + str(round(elitist_complexity.std(), 2))
        values += "," + str(elitist_complexity.median())

        # Calculates both mse and std of mse between all runs (Changes MSE to positive value)
        mse = -fold_df['test_neg_mean_squared_error']
        values += "," + str(round(mse.mean(), 4))
        values += "," + str(round(mse.std(), 4))

        values += '\n\n'

    print(f"{directory} FINISHED")
    with open(f"../{directory}/Results.csv", "w") as file:
        file.write(header + values)
