import os
import csv
import numpy as np


def get_rows(input_directory: str, experiment_name: str, dataset_name: str, file_line_func_name: str):
    row_list = []
    all_experiments = [item for item in next(os.walk(input_directory))[1]]
    if experiment_name in all_experiments:
        all_datasets = [item for item in next(os.walk(f"{input_directory}/{experiment_name}"))[1]]
        if dataset_name in all_datasets:
            for data_file in os.listdir(f"{input_directory}/{experiment_name}/{dataset_name}"):
                with open(f"{input_directory}/{experiment_name}/{dataset_name}/{data_file}") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if file_line_func_name in row["filename:lineno(function)"]:
                            row_list.append(row)
    return row_list


def print_mean_stdvar(
    input_directory: str, experiment_name: str, dataset_name: str, file_line_func_name: str, attribute: str
):
    row_list = get_rows(input_directory, experiment_name, dataset_name, file_line_func_name)
    attribute_list = [float(item[attribute]) for item in row_list]
    attribute_list = np.asarray(attribute_list)
    mean = np.mean(attribute_list)
    stdv = np.std(attribute_list)
    print(f"{experiment_name}, {dataset_name}, {attribute}, {mean}, {stdv}")


dataset_list = [
    "combined_cycle_power_plant",
    "airfoil_self_noise",
    "concrete_strength",
    "parkinson_total",
    "protein_structure",
]
optimizer_list = ["GA", "SAGA1", "SAGA2", "SAGA3"]

if __name__ == "__main__":
    for dataset in dataset_list:
        print("==============================================")
        for optimizer in optimizer_list:
            print_mean_stdvar("profiling_data", optimizer, dataset, "suprb.py:96(fit)", "cumtime")
