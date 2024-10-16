import json
import os
import pandas as pd
from tabulate import tabulate
from logging_output_scripts.utils import check_and_create_dir
import utils

"""
This script uses the tabulate package (https://pypi.org/project/tabulate/)
to create LaTex tables based on the values calculated in Summary_csv.py
(Except for Genomes-Tables which use a Json)
"""
with open('logging_output_scripts/config.json') as f:
    config = json.load(f)

final_output_dir = f"{config['output_directory']}"
summary_csv_dir = f"{final_output_dir}/csv_summary"

# Empty string needed for formatting purposes

comp_column = {0: 'MEAN_COMP', 1: 'STD_COMP', 2: 'MEDIAN_COMP', 3: 'MIN_COMP', 4: 'MAX_COMP'}
comp_column_short = {0: 'mean', 1: 'standard deviation', 2: 'median', 3: 'min', 4: 'max'}


# Returns column with name "column_name" of "problem"


def load_problem_columns(df, column_name):
    data_res = []
    for problem in config["datasets"]:
        res = df[df['Problem'].str.contains(problem)]
        data_res.append(float(res[column_name]))
    return data_res


def create_folder(folder_name):
    directory = folder_name
    if not os.path.exists(directory):
        os.mkdir(directory)

# COMPLEXITY TABLES


def write_complexity_all(dataset_shorts):
    comp_list = []
    for heuristic in config["heuristics"]:
        df = pd.read_csv(f"{summary_csv_dir}/{config['heuristics'][heuristic]}_summary.csv")
        for column_name, column_name_short in zip(comp_column.values(), comp_column_short.values()):
            data_res = load_problem_columns(df, column_name)
            comp_list.append((config["heuristics"][heuristic], column_name_short, data_res[0], data_res[1], data_res[2], data_res[3]))

    res = tabulate(comp_list, tablefmt="latex_booktabs", headers=dataset_shorts.values())

    with open(f"{final_output_dir}/latex_tabular/ComplexityAll.txt", "w") as file:
        file.write(res)

# COMPLEXITY TABLES


def write_complexity():
    for heuristic in config["heuristics"]:
        df = pd.read_csv(f"{summary_csv_dir}/{heuristic}_summary.csv")
        comp_list = []
        for column_name, column_name_short in zip(comp_column.values(), comp_column_short.values()):
            data_res = load_problem_columns(df, column_name)
            if heuristic == "ES":
                heuristic = "Suprb"
            comp_list.append((column_name_short, data_res[0], data_res[1], data_res[2], data_res[3]))

        res = tabulate(comp_list, tablefmt="latex_booktabs", headers=datasets_short.values())

        with open(f"{final_output_dir}/Complexity-{heuristic}.txt", "w") as file:
            file.write(res)

# GENOMES (Requires Genomes to be Stored as a .json in the respective Folder) [Leave out if not needed]


def write_genomes():
    for heuristic in config["heuristics"]:
        for problem in config["datasets"]:
            with open(f'Genomes/{heuristic}_{problem}.json', 'r') as f:
                genomes = json.load(f)

            genomes = list(genomes.values())

            genome_list = []
            # Create tuple of shape (iteration, genome[iteration])
            for x in range(len(genomes)):
                genome_list.append((str(x), genomes[x]))

            headers = ['Iteration', 'Genome']
            genome_table = tabulate(genome_list, tablefmt="latex_booktabs", headers=headers)

            create_folder("Genomes")

            with open(f"Genomes/{heuristic}-{problem}.txt", "w") as file:
                file.write(genome_table)


def write_mse_all(datasets_short):
    """
    Creates tables of shape:
             [Problem i]
    [MODEL 1]     ...
    [MODEL 2]     ...
        ...
    [MODEL n]
    To be fused into a larger table (needs to be done manually)
    """
    # Each column features all models for one problem
    column = []
    row = []
    for problem in config["datasets"]:
        # Each row features one problem for one model
        for heuristic in config["heuristics"]:
            df = pd.read_csv(f"{summary_csv_dir}/{config['heuristics'][heuristic]}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(((round(float(res['MEAN_MSE']), len(config["datasets"]))), round(float(res['STD_MSE']), 5)))
        column.append(row)

    res = tabulate(column, tablefmt="latex_booktabs", headers=datasets_short.values())

    with open(f"{final_output_dir}/latex_tabular/MSEAll.txt", "w") as file:
        file.write(res)


# MSE TABLES (Creates individual tables to be combined into a larger tabular)
def write_mse():
    """
    Creates tables of shape:
             [Problem i]
    [MODEL 1]     ...
    [MODEL 2]     ...
        ...
    [MODEL n]
    To be fused into a larger table (needs to be done manually)
    """
    # Each column features all models for one problem
    column = []
    for problem in config["datasets"]:
        # Each row features one problem for one model
        row = []
        for heuristic, renamed_heuristic in config['heuristics'].items():
            df = pd.read_csv(f"{summary_csv_dir}/{renamed_heuristic}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(((round(float(res['MEAN_MSE']), 4)), round(float(res['STD_MSE']), 4)))
        column.append(row)

    headers = ['MSE', 'STD']
    problem_1 = tabulate(column[0], tablefmt="latex_booktabs", headers=headers)
    problem_2 = tabulate(column[1], tablefmt="latex_booktabs", headers=headers)
    problem_3 = tabulate(column[2], tablefmt="latex_booktabs", headers=headers)
    problem_4 = tabulate(column[3], tablefmt="latex_booktabs", headers=headers)

    with open(f"{final_output_dir}/latex_tabular/MSE.txt", "w") as file:
        file.write(problem_1 + "\n\n" + problem_2 + "\n\n" + problem_3 + "\n\n" + problem_4)


def single_table(dataset_shorts):
    columns = []
    for problem in config["datasets"]:
        # Each row features one problem for one model
        row = []
        row.append(problem)
        for heuristic in config["heuristics"]:
            df = pd.read_csv(f"{summary_csv_dir}/{heuristic}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(str(round(float(res['MEAN_MSE']), 2))+u"\u00B1" +
                       str(round(float(res['STD_MSE']), 2)))
            row.append(str(round(float(res['MEAN_COMP']), 2))+u"\u00B1" +
                       str(round(float(res['STD_COMP']), 2)))
        columns.append(row)
    frame = pd.DataFrame(columns)
    headers = [x for y in [['MSE', 'Complexity'] for i in range(
        frame.shape[1]-1)] for x in y]
    latex = tabulate(frame, tablefmt="latex_booktabs", headers=datasets_short.values())
    splits = latex.split("\\toprule")
    methods = " ".join(["\\multicolumn{2}{c}{"+h+"} &" for h in config["heuristics"]])
    latex = splits[0]+"\\toprule"+methods+splits[1]
    with open(f"{final_output_dir}/combined.txt", "w") as file:
        file.write(latex)

# TODO: Vor +- muss ein &


def single_table_all_mse(dataset_shorts):
    columns = []
    for heuristic in config["heuristics"]:
        # Each row features one problem for one model
        row = [utils.datasets_map[heuristic]]
        for problem in config["datasets"]:
            df = pd.read_csv(f"{summary_csv_dir}/{config['heuristics'][heuristic]}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(str(round(float(res['MEAN_MSE']), 2))+u"\u00B1" +
                       str(round(float(res['STD_MSE']), 2)))
        columns.append(row)
    frame = pd.DataFrame(columns)
    headers = [x for y in [['MSE'] for i in range(frame.shape[1]-1)] for x in y]
    headers = [i for i in range(frame.shape[1]-1)]
    latex = tabulate(frame, tablefmt="latex_booktabs", headers=headers)
    splits = latex.split("\\toprule")
    methods = " ".join(["\\multicolumn{2}{c}{"+h+"} &" for h in dataset_shorts.values()])
    print(methods)
    latex = splits[0]+"\\toprule"+methods+splits[1]
    with open(f"{final_output_dir}/latex_tabular/latex_mse.txt", "w") as file:
        file.write(latex)


def single_table_all_complexity():
    columns = []
    for heuristic in config["heuristics"]:
        # Each row features one problem for one model
        row = [utils.datasets_map[heuristic]]
        for problem in config["datasets"]:
            df = pd.read_csv(f"{summary_csv_dir}/{config['heuristics'][heuristic]}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(str(round(float(res['MEAN_COMP']), 2))+u"\u00B1" +
                       str(round(float(res['STD_COMP']), 2)))
        columns.append(row)
    frame = pd.DataFrame(columns)
    headers = [x for y in [['Complexity'] for i in range(
        frame.shape[1]-1)] for x in y]
    headers = [i for i in range(frame.shape[1]-1)]
    latex = tabulate(frame, tablefmt="latex_booktabs", headers=headers)
    splits = latex.split("\\toprule")
    methods = " ".join(["\\multicolumn{2}{c}{"+h+"} &" for h in config["datasets"]])
    latex = splits[0]+"\\toprule"+methods+splits[1]
    with open(f"{final_output_dir}/latex_tabular/latex_complexity.txt", "w") as file:
        file.write(latex)


# Add / leave out certain tables
if __name__ == '__main__':
    final_output_dir = f"{config['output_directory']}"
    check_and_create_dir(final_output_dir, 'latex_tabular')
    # write_complexity()
    # write_complexity_all()
    # Only use if genomes are actually tracked.
    # write_genomes()
    write_mse()
    # write_mse_all()
    # single_table()
    # single_table_all_mse()
    # single_table_all_complexity()
