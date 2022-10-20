import json
import os
from tabulate import tabulate
import pandas as pd

"""
This script uses the tabulate package (https://pypi.org/project/tabulate/)
to create LaTex tables based on the values calculated in Summary_csv.py
(Except for Genomes-Tables which use a Json)
"""

path_to_csvs = r"C:\Users\m\Documents\SupRB\rule_discovery_paper\run_csvs"
tables = r"C:\Users\m\Documents\SupRB\rule_discovery_paper\tables"

heur = ['ES', 'RS', 'NS', 'MCNS', 'NSLC']

datasets = ["concrete_strength", 'combined_cycle_power_plant',
            'airfoil_self_noise', 'energy_cool']

# Empty string needed for formatting purposes
datasets_short = {-1: ' ', 0: 'CS', 1: 'CCPP', 2: 'ASN', 3: 'EC'}

comp_column = {0: 'MEAN_COMP', 1: 'STD_COMP', 2: 'MEDIAN_COMP', 3: 'MIN_COMP', 4: 'MAX_COMP'}
comp_column_short = {0: 'mean', 1: 'standard deviation', 2: 'median', 3: 'min', 4: 'max'}


# Returns column with name "column_name" of "problem"
def load_problem_columns(df, column_name):
    data_res = []
    for problem in datasets:
        res = df[df['Problem'].str.contains(problem)]
        data_res.append(float(res[column_name]))
    return data_res


def create_folder(folder_name):
    directory = folder_name
    if not os.path.exists(directory):
        os.mkdir(directory)


# COMPLEXITY TABLES
def write_complexity():
    for directory in heur:
        df = pd.read_csv(f"{path_to_csvs}/{directory}/summary.csv")
        comp_list = []
        for column_name, column_name_short in zip(comp_column.values(), comp_column_short.values()):
            data_res = load_problem_columns(df, column_name)
            comp_list.append((column_name_short, data_res[0], data_res[1], data_res[2], data_res[3]))

        res = tabulate(comp_list, tablefmt="latex_booktabs", headers=datasets_short.values())

        with open(f"{tables}/Complexity-{directory}.txt", "w") as file:

            file.write(res)

# GENOMES (Requires Genomes to be Stored as a .json in the respective Folder) [Leave out if not needed]
def write_genomes():
    for directory in heur:
        for problem in datasets:
            with open(f'Genomes/{directory}_{problem}.json', 'r') as f:
                genomes = json.load(f)

            genomes = list(genomes.values())

            genome_list = []
            # Create tuple of shape (iteration, genome[iteration])
            for x in range(len(genomes)):
                genome_list.append((str(x), genomes[x]))

            headers = ['Iteration', 'Genome'
            genome_table = tabulate(genome_list, tablefmt="latex_booktabs", headers=headers)

            create_folder("Genomes")

            with open(f"Genomes/{directory}-{problem}.txt", "w") as file:
                file.write(genome_table)


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
    for problem in datasets:
        # Each row features one problem for one model
        row = []
        for directory in heur:
            df = pd.read_csv(f"{path_to_csvs}/{directory}/summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(((round(float(res['MEAN_MSE']), 4)), round(float(res['STD_MSE']), 4)))
        column.append(row)

    headers = ['MSE', 'STD']
    problem_1 = tabulate(column[0], tablefmt="latex_booktabs", headers=headers)
    problem_2 = tabulate(column[1], tablefmt="latex_booktabs", headers=headers)
    problem_3 = tabulate(column[2], tablefmt="latex_booktabs", headers=headers)
    problem_4 = tabulate(column[3], tablefmt="latex_booktabs", headers=headers)

    with open(f"{tables}/MSE.txt", "w") as file:
        file.write(problem_1 + "\n\n" + problem_2 + "\n\n" + problem_3 + "\n\n" + problem_4)


def single_table():
    columns = []
    for problem in datasets:
        # Each row features one problem for one model
        row = []
        row.append(problem)
        for directory in heur:
            df = pd.read_csv(f"{path_to_csvs}/{directory}/summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(str(round(float(res['MEAN_MSE']), 2))+u"\u00B1"+
                       str(round(float(res['STD_MSE']), 2)))
            row.append(str(round(float(res['MEAN_COMP']), 2))+u"\u00B1"+
                       str(round(float(res['STD_COMP']), 2)))
        columns.append(row)
    frame = pd.DataFrame(columns)
    headers = [x for y in [['MSE', 'Complexity'] for i in range(
        frame.shape[1]-1)] for x in y ]
    latex = tabulate(frame, tablefmt="latex_booktabs", headers=headers)
    splits = latex.split("\\toprule")
    methods = " ".join(["\\multicolumn{2}{c}{"+h+"} &" for h in heur])
    latex = splits[0]+"\\toprule"+methods+splits[1]
    with open(f"{tables}/combined.txt", "w") as file:
        file.write(latex)


# Add / leave out certain tables
if __name__ == '__main__':
    # write_complexity()
    # Only use if genomes are actually tracked.
    # write_genomes()
    # write_mse()
    single_table()
