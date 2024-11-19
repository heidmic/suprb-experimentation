import mlflow
import numpy as np
import pandas as pd

# all_runs_df = mlflow.search_runs(search_all_experiments=True)

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"

# df = all_runs_df[(all_runs_df["tags.fold"] == 'True')]
# df = df[["tags.mlflow.runName", mse, complexity]]
# print(round(np.min(df[mse]), round(np.max(df[mse]), round(np.min(df[complexity]), round(np.max(df[complexity])),5)

# df[mse] *= -1

# df.to_csv(f"adeles_suprb.csv", index=False)
# print(df)


all_df = pd.read_csv(f"adeles_suprb.csv")

runs = [862288241,
        1184028159,
        1691623607,
        3581440988,
        2046530742,
        2669555309,
        3444837047,
        2099784219]

# Print header
print(f"{'Min MSE':<20} {'Max MSE':<20} {'Mean MSE':<20} {'Std MSE':<20} "
      f"{'Min Complexity':<15} {'Max Complexity':<15} {'Mean Complexity':<15} {'Std Complexity':<15}")

# Print each row with formatted output
df = all_df[all_df["tags.mlflow.runName"].isin(runs)]

# df = all_df[all_df["tags.mlflow.runName"] == run_id]
min_mse = round(np.min(df[mse]), 5)
max_mse = round(np.max(df[mse]), 5)
mean_mse = round(np.mean(df[mse]), 5)
std_mse = round(np.std(df[mse]), 5)

min_complexity = round(np.min(df[complexity]), 5)
max_complexity = round(np.max(df[complexity]), 5)
mean_complexity = round(np.mean(df[complexity]), 5)
std_complexity = round(np.std(df[complexity]), 5)

# Print each row with consistent spacing and precision
print(f"{min_mse:<20.10f} {max_mse:<20.10f} {mean_mse:<20.10f} {std_mse:<20.10f} "
      f"{min_complexity:<15.1f} {max_complexity:<15.1f} {mean_complexity:<15.3f} {std_complexity:<15.10f}")


# Define your decision_tree_unlimited_depth data
decision_tree_unlimited_depth = {
    '1': [[0.00017219, 0.00023481, 0.0001912, 0.00020447, 0.00034125, 0.00020159, 0.00025836, 0.00014216], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
    '2': [[0.0001812, 0.00024049, 0.00018722, 0.00020631, 0.00033783, 0.00020162, 0.00022029, 0.00014205], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
    '3': [[0.00017379, 0.00023236, 0.00018744, 0.00020444, 0.00032908, 0.00020091, 0.00023222, 0.00013181], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
    '4': [[0.00035573, 0.00023238, 0.00018644, 0.00020448, 0.00033624, 0.00020097, 0.00024075, 0.00012694], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
    '5': [[0.00017137, 0.00023806, 0.00019419, 0.00020636, 0.00033804, 0.00020081, 0.00021456, 0.00013155], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
    '6': [[0.00017268, 0.00023405, 0.00018754, 0.00020453, 0.00032888, 0.00020195, 0.00024164, 0.00014191], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
    '7': [[0.00018166, 0.00023416, 0.00018396, 0.00020731, 0.00033801, 0.00020187, 0.00024574, 0.00013486], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
    '8': [[0.00018101, 0.00023417, 0.00018991, 0.00020724, 0.00034116, 0.00020067, 0.00023226, 0.00014188], [48, 45, 45, 44, 45, 43, 44, 44], [17480, 17478, 17461, 17500, 17495, 17442, 17522, 17525]],
}


# Print header
print("\n\n---------------------------------------------------------------------")
print("Depth: Unlimted")
print(f"{'Min MSE':<20} {'Max MSE':<20} {'Mean MSE':<20} {'Std MSE':<20} "
      f"{'Min Depth':<15} {'Max Depth':<15} {'Mean Depth':<15} {'Std Depth':<15} "
      f"{'Min Leaves':<15} {'Max Leaves':<15} {'Mean Leaves':<15} {'Std Leaves':<15}")

all_mse = [value for sublist in [v[0] for v in decision_tree_unlimited_depth.values()] for value in sublist]
all_complexity = [value for sublist in [v[1] for v in decision_tree_unlimited_depth.values()] for value in sublist]
all_leaves = [value for sublist in [v[2] for v in decision_tree_unlimited_depth.values()] for value in sublist]


# Iterate over each dictionary in the DataFrame and calculate statistics
min_mse = round(np.min(all_mse), 5)
max_mse = round(np.max(all_mse), 5)
mean_mse = round(np.mean(all_mse), 5)
std_mse = round(np.std(all_mse), 5)

min_complexity = round(np.min(all_complexity), 5)
max_complexity = round(np.max(all_complexity), 5)
mean_complexity = round(np.mean(all_complexity), 5)
std_complexity = round(np.std(all_complexity), 5)

min_leaves = round(np.min(all_leaves), 5)
max_leaves = round(np.max(all_leaves), 5)
mean_leaves = round(np.mean(all_leaves), 5)
std_leaves = round(np.std(all_leaves), 5)

# Print each row with consistent spacing and precision
print(f"{min_mse:<20.10f} {max_mse:<20.10f} {mean_mse:<20.10f} {std_mse:<20.10f} "
      f"{min_complexity:<15.1f} {max_complexity:<15.1f} {mean_complexity:<15.3f} {std_complexity:<15.10f} "
      f"{min_leaves:<15.1f} {max_leaves:<15.1f} {mean_leaves:<15.3f} {std_leaves:<15.10f}")


# Define your decision_tree_unlimited_depth data
decision_tree_depth_5 = {
    '1': [0.03359901, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284137, 0.03311119, 0.03369361],
    '2': [0.0335993, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284166, 0.03310182, 0.03369391],
    '3': [0.03359901, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284137, 0.03310359, 0.03369361],
    '4': [0.03359901, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284137, 0.03311119, 0.03369391],
    '5': [0.0335993, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284137, 0.03310359, 0.03369391],
    '6': [0.03359901, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284137, 0.03310182, 0.03369361],
    '7': [0.03359901, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284137, 0.03311119, 0.03369391],
    '8': [0.03359901, 0.0342611, 0.03327596, 0.03313299, 0.03352422, 0.03284137, 0.03311119, 0.03369391]
}

# Print header
print("\n\n---------------------------------------------------------------------")
print("Depth: 5")
print(f"{'Min MSE':<20} {'Max MSE':<20} {'Mean MSE':<20} {'Std MSE':<20}")

all_mse = [value for sublist in decision_tree_depth_5.values() for value in sublist]

# Calculate statistics
min_mse = round(np.min(all_mse), 5)
max_mse = round(np.max(all_mse), 5)
mean_mse = round(np.mean(all_mse), 5)
std_mse = round(np.std(all_mse), 5)

# Print each row with consistent spacing and precision
print(f"{min_mse:<20.10f} {max_mse:<20.10f} {mean_mse:<20.10f} {std_mse:<20.10f}")
