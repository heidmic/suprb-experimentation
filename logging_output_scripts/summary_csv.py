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

for i in dirs:
    # Head of csv-File
    s = f"Problem,MEAN_COMP,STD_COMP,MEDIAN_COMP,MIN_COMP,MAX_COMP,MEAN_MSE,STD_MSE," \
        f"FIN_ITER_MAX,FIN_ITER_MIN,FIN_ITER_MEAN,THRESH_0_MEAN,THRESH_0_MIN,THRESH_0_MAX," \
        f"THRESH_1_MEAN,THRESH_1_MIN,THRESH_1_MAX,THRESH_2_MEAN,THRESH_2_MIN,THRESH_2_MAX"
    # Current working directory
    directory = dirs[i]

    for j in datasets:
        problem = datasets[j]
        print(f"WORKING ON DATASET {problem} WITH {directory}")
        # Read from csv-File in directory named after model and named after dataset
        df = pd.read_csv(f"../{directory}/{problem}.csv")
        # Filter out individual runs (Removes averaged values)
        fold_df = df[df['Name'].str.contains('fold')]

        # Calculates mean, min, max, median and std of elitist_complexity across all runs
        elitist_complexity = fold_df['elitist_complexity']
        min_comp = elitist_complexity.min()
        max_comp = elitist_complexity.max()
        mean_comp = elitist_complexity.mean()
        std_comp = elitist_complexity.std()
        median_comp = elitist_complexity.median()

        # Calculates both mse and std of mse between all runs (Changes MSE to positive value)
        mse = -fold_df['test_neg_mean_squared_error']
        mean_mse = mse.mean()
        std_mse = mse.std()

        # Calculates the amount of iterations during rule discovery (mean, max, min)
        mean_final_iter = fold_df['delay_mean'].mean()
        max_final_iter = fold_df['delay_max'].max()
        min_final_iter = fold_df['delay_min'].min()

        # Calculates the iterations the elitist remained unchanged (t = 0, 1, 2)
        thresh_0 = fold_df['elitist_convergence_thresh_0']
        thresh_1 = fold_df['elitist_convergence_thresh_1']
        thresh_2 = fold_df['elitist_convergence_thresh_2']

        mean_t0 = thresh_0.mean()
        max_t0 = thresh_0.max()
        min_t0 = thresh_0.min()
        std_t0 = thresh_0.std()

        mean_t1 = thresh_1.mean()
        max_t1 = thresh_1.max()
        min_t1 = thresh_1.min()
        std_t1 = thresh_1.std()

        mean_t2 = thresh_2.mean()
        max_t2 = thresh_2.max()
        min_t2 = thresh_2.min()
        std_t2 = thresh_2.std()

        # Appends row of results for each Dataset
        s += f"\n\n{problem},{mean_comp},{std_comp},{median_comp},{min_comp},\
        {max_comp},{mean_mse},{std_mse},{max_final_iter},{min_final_iter},\
        {mean_final_iter},{mean_t0},{min_t0},{max_t0},{mean_t1},{min_t1},\
        {max_t1},{mean_t2},{min_t2},{max_t2}"

    print(f"{directory} FINISHED")
    with open(f"../{directory}/Results.csv", "w") as file:
        file.write(s)
