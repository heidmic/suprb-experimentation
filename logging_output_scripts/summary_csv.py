import pandas as pd

"""
Extracts values from csv-Files gained from mlflow, performs calculations and
stores the results in a new csv-File for all Models specified
"""

# Datasets runs were performed on, responds to one csv file each
datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}
# The used representations (or models) on which runs were performed
dirs = {0: 'OBR', 1: 'UBR', 2: 'CSR', 3: 'MPR'}

for directory in dirs.values():
    # Head of csv-File
    header = f"Problem,MEAN_COMP,STD_COMP,MEDIAN_COMP,MIN_COMP,MAX_COMP,MEAN_MSE,STD_MSE," \
             f"FIN_ITER_MAX,FIN_ITER_MIN,FIN_ITER_MEAN,THRESH_0_MEAN,THRESH_0_MIN,THRESH_0_MAX," \
             f"THRESH_1_MEAN,THRESH_1_MIN,THRESH_1_MAX,THRESH_2_MEAN,THRESH_2_MIN,THRESH_2_MAX"
    values = ""
    for problem in datasets.values():
        values = problem

        print(f"WORKING ON DATASET {problem} WITH {directory}")
        # Read from csv-File in directory named after model and named after dataset
        df = pd.read_csv(f"../{directory}/{problem}.csv")
        # Filter out individual runs (Removes averaged values)
        fold_df = df[df['Name'].str.contains('fold')]

        # Calculates mean, min, max, median and std of elitist_complexity across all runs
        elitist_complexity = fold_df['elitist_complexity']
        values += "," + elitist_complexity.min()
        values += "," + elitist_complexity.max()
        values += "," + elitist_complexity.mean()
        values += "," + elitist_complexity.std()
        values += "," + elitist_complexity.median()

        # Calculates both mse and std of mse between all runs (Changes MSE to positive value)
        mse = -fold_df['test_neg_mean_squared_error']
        values += "," + mse.mean()
        values += "," + mse.std()

        # Calculates the amount of iterations during rule discovery (mean, max, min)
        values += "," + fold_df['delay_mean'].mean()
        values += "," + fold_df['delay_max'].max()
        values += "," + fold_df['delay_min'].min()

        # Calculates the iterations the elitist remained unchanged (t = 0, 1, 2)
        thresh_0 = fold_df['elitist_convergence_thresh_0']
        thresh_1 = fold_df['elitist_convergence_thresh_1']
        thresh_2 = fold_df['elitist_convergence_thresh_2']

        values += "," + thresh_0.mean()
        values += "," + thresh_0.max()
        values += "," + thresh_0.min()
        values += "," + thresh_0.std()

        values += "," + thresh_1.mean()
        values += "," + thresh_1.max()
        values += "," + thresh_1.min()
        values += "," + thresh_1.std()

        values += "," + thresh_2.mean()
        values += "," + thresh_2.max()
        values += "," + thresh_2.min()
        values += "," + thresh_2.std()

        values += '\n\n'

    print(f"{directory} FINISHED")
    with open(f"../{directory}/Results.csv", "w") as file:
        file.write(header)
