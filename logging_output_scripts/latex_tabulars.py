import json
from tabulate import tabulate
import pandas as pd

"""
This script uses tabulate package to create LaTex tables based on the values
calculated in Summary_csv.py (Except for Genomes-Tables which use a Json)
"""

dirs = {0: 'OBR', 1: 'UBR', 2: 'CSR', 3: 'MPR'}

# *_short Dictionaries are used in the tables headers whereas other dicts are used to access data from file
datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}

# Empty string needed for formatting purposes
datasets_short = {-1: ' ', 0: 'PT', 1: 'PTTS', 2: 'ASN',
                  3: 'CS', 4: 'CCPP'}

comp_column = {0: 'MEAN_COMP', 1: 'STD_COMP', 2: 'MEDIAN_COMP', 3: 'MIN_COMP', 4: 'MAX_COMP'}
comp_column_short = {0: 'mean', 1: 'standard deviation', 2: 'median', 3: 'min', 4: 'max'}

thresh_column = {0: 'THRESH_0_MEAN', 1: 'THRESH_0_MIN', 2: 'THRESH_0_MAX', 3: 'THRESH_1_MEAN', 4: 'THRESH_1_MIN',
                 5: 'THRESH_1_MAX', 6: 'THRESH_2_MEAN', 7: 'THRESH_2_MIN', 8: 'THRESH_2_MAX'}
thresh_column_short = {0: 'mean', 1: 'min', 2: 'max', 3: 'mean', 4: 'min',
                       5: 'max', 6: 'mean', 7: 'min', 8: 'max'}

iter_column = {0: 'FIN_ITER_MAX', 1: 'FIN_ITER_MIN', 2: 'FIN_ITER_MEAN'}
iter_column_short = {0: 'max', 1: 'min', 2: 'mean'}


# Returns column with name "column_name" of "problem"
def load_problem_columns(df, column_name):
    data_res = []
    for problem in datasets.values():
        res = df[df['Problem'].str.contains(problem)]
        data_res.append(float(res[column_name]))
        return data_res


# COMPLEXITY TABLES
def write_complexity():
    for directory in dirs.values():
        df = pd.read_csv(f"../{directory}/Results.csv")
        comp_list = []
        for column_name, column_name_short in zip(comp_column.values(), comp_column_short.values()):
            data_res = load_problem_columns(df, column_name)
            comp_list.append((column_name_short, data_res[0], data_res[1], data_res[2], data_res[3], data_res[4]))

        res = tabulate(comp_list, tablefmt="latex", headers=datasets_short.values())
        with open(f"Complexity/Complexity-{directory}.txt", "w") as file:
            file.write(res)


# GENOMES
def write_genomes():
    for directory in dirs.values():
        for problem in datasets.values():
            with open(f'Genomes/{directory}_{problem}.json', 'r') as f:
                genomes = json.load(f)

            genomes = list(genomes.values())

            genome_list = []
            # Create tuple of shape (iteration, genome[iteration])
            for x in range(len(genomes)):
                genome_list.append((str(x), genomes[x]))

            headers = ['Iteration', 'Genome']
            genome_table = tabulate(genome_list, tablefmt="latex", headers=headers)
            with open(f"Genomes/{directory}-{problem}.txt", "w") as file:
                file.write(genome_table)


# MSE TABLES (Creates individual tables to be combined into a larger tabular)
def write_mse():
    """
    Creates table of shape:
            [Problem i]
    [REP 1]     ...
    [REP 2]     ...
    ...
    To be fused into a larger table
    """
    # Each column features all models for one problem
    column = []
    for problem in datasets.values():
        # Each row features one problem for one model
        row = []
        for directory in dirs.values():
            df = pd.read_csv(f"../{directory}/Results.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(((round(float(res['MEAN_MSE']), 4)), round(float(res['STD_MSE']), 4)))
        column.append(row)

    headers = ['MSE', 'STD']
    table_1 = tabulate(column[0], tablefmt="latex", headers=headers)
    table_2 = tabulate(column[1], tablefmt="latex", headers=headers)
    table_3 = tabulate(column[2], tablefmt="latex", headers=headers)
    table_4 = tabulate(column[3], tablefmt="latex", headers=headers)
    table_5 = tabulate(column[4], tablefmt="latex", headers=headers)

    with open(f"MSE/MSE.txt", "w") as file:
        file.write(table_1 + "\n\n" + table_2 + "\n\n" + table_3 + "\n\n" + table_4 + "\n\n" + table_5)


# CONVERGENCE TABLES
def write_convergence():
    for directory in dirs.values():
        df = pd.read_csv(f"../{directory}/Results.csv")
        comp_list = []
        for column_index in thresh_column:
            column_name = thresh_column[column_index]
            data_res = load_problem_columns(df, column_name)
            threshold_string = " "
            if column_index % 3 == 0:
                threshold = int(column_index / 3)
                threshold_string = f'Threshold = {threshold}'

            comp_list.append((threshold_string, thresh_column_short[column_index], data_res[0],
                              data_res[1], data_res[2], data_res[3], data_res[4]))

        new_header = list(datasets_short.values())
        new_header.insert(0, " ")
        convergence_table = tabulate(comp_list, tablefmt="latex", headers=new_header)
        with open(f"Convergence/convergence-{directory}.txt", "w") as file:
            file.write(convergence_table)


# Final iter TABLES
def write_final_iter():
    for directory in dirs.values():
        df = pd.read_csv(f"../{directory}/Results.csv")
        comp_list = []
        for column_name, column_name_short in zip(iter_column.values(), iter_column_short.values()):
            data_res = load_problem_columns(df, column_name)
            comp_list.append((column_name_short, data_res[0], data_res[1],
                              data_res[2], data_res[3], data_res[4]))

        final_iter_table = tabulate(comp_list, tablefmt="latex", headers=datasets_short.values())
        with open(f"Iter/iter-{directory}.txt", "w") as file:
            file.write(final_iter_table)


# Add / leave out certain tables
if __name__ == '__main__':
    write_complexity()
    write_genomes()
    write_convergence()
    write_mse()
    write_final_iter()
