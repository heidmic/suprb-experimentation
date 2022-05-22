import json
import numpy as np
import scipy
from tabulate import tabulate
import pandas as pd

"""
This script uses tabulate package to create LaTex tables based on the values
calculated in Summary_csv.py (Except for Genomes-Tables which use a Json)
"""


dirs = {0 : 'OBR', 1 : 'UBR', 2 : 'CSR', 3 : 'MPR'}
datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}
datasets_short = {-1: ' ', 0: 'PT', 1: 'PTTS', 2: 'ASN',
            3: 'CS', 4: 'CCPP'}
comp_column = {0 : 'MEAN_COMP', 1 : 'STD_COMP', 2 : 'MEDIAN_COMP', 3 : 'MIN_COMP', 4: 'MAX_COMP'}
comp_column_long = {0 : 'mean', 1 : 'standard deviation', 2 : 'median', 3 : 'min', 4: 'max'}

thresh_column = {0 : 'THRESH_0_MEAN', 1 : 'THRESH_0_MIN'  ,2 : 'THRESH_0_MAX',	3 : 'THRESH_1_MEAN', 4 : 'THRESH_1_MIN',
	5 : 'THRESH_1_MAX',	6 : 'THRESH_2_MEAN', 7 : 'THRESH_2_MIN', 8 : 'THRESH_2_MAX'}
thresh_column_long = {0 : 'mean', 1 : 'min'  ,2 : 'max',	3 : 'mean', 4 : 'min',
	5 : 'max',	6 : 'mean', 7 : 'min', 8 : 'max'}

iter_column = {0 : 'FIN_ITER_MAX', 1 : 'FIN_ITER_MIN', 2 : 'FIN_ITER_MEAN'}
iter_column_long = {0 : 'max', 1 : 'min', 2 : 'mean'}

# COMPLEXITY TABLES
for i in dirs:
    dir = dirs[i]
    df = pd.read_csv(f"../{dir}/Results.csv")

    file = open(f"Complexity/Complexity-{dir}.txt", "w")
    comp_list = []
    for i in comp_column:
        c = comp_column[i]
        temp = []
        for v in datasets.values():
            res = df[df['Problem'].str.contains(v)]
            print(f"Current series : {res[c]} Dataset {v} and Rep {dir}")
            temp.append(float(res[c]))
        comp_list.append((comp_column_long[i], temp[0], temp[1], temp[2], temp[3], temp[4]))

    res = tabulate(comp_list, tablefmt="latex", headers=datasets_short.values())
    file.write(res)

# Genomes
for i in dirs:
    dir = dirs[i]
    for j in datasets:
        dataset = datasets[j]
        if j == 2 or dir == 'CSR' or dir == 'MPR':
            break

        with open(f'Genomes/{dir}_{dataset}.json', 'r') as f:
            genomes = json.load(f)

        genomes = list(genomes.values())

        file = open(f"Genomes/{dataset}-{dir}.txt", "w")
        genome_list = []
        for i in range(len(genomes)):
            temp = genomes[i]
            genome_list.append((str(i), temp))

        headers = ['Iteration', 'Genome']
        res = tabulate(genome_list, tablefmt="latex", headers=headers)
        file.write(res)

# MSE TABLES (Creates individual tables to be combined into a larger tabular)
row = []
file2 = open(f"MSE/MSE.txt", "w")
for v in datasets.values():
    temp = []
    for i in dirs:
        dir = dirs[i]
        df = pd.read_csv(f"../{dir}/Results.csv")
        res = df[df['Problem'].str.contains(v)]
        temp.append(((round(float(res['MEAN_MSE']), 4)), round(float(res['STD_MSE']), 4)))
    row.append(temp)

table_1 = tabulate(row[0], tablefmt="latex", headers=['MSE', 'STD'])
table_2 = tabulate(row[1], tablefmt="latex", headers=['MSE', 'STD'])
table_3 = tabulate(row[2], tablefmt="latex", headers=['MSE', 'STD'])
table_4 = tabulate(row[3], tablefmt="latex", headers=['MSE', 'STD'])
table_5 = tabulate(row[4], tablefmt="latex", headers=['MSE', 'STD'])
file2.write(table_1 + "\n\n" + table_2 + "\n\n" + table_3 + "\n\n" + table_4 + "\n\n" + table_5)

# CONVERGENCE TABLES
for i in dirs:
    dir = dirs[i]
    df = pd.read_csv(f"../{dir}/Results.csv")

    file = open(f"Convergence/convergence-{dir}.txt", "w")
    comp_list = []
    for i in thresh_column:
        c = thresh_column[i]
        temp = []
        for v in datasets.values():
            res = df[df['Problem'].str.contains(v)]
            temp.append(float(res[c]))
        if i % 3 == 0:
            threshhold = int(i / 3)
            comp_list.append((f'Threshhold = {threshhold}',thresh_column_long[i], temp[0], temp[1], temp[2], temp[3], temp[4]))
        else:
            comp_list.append((" " ,thresh_column_long[i], temp[0], temp[1], temp[2], temp[3], temp[4]))
    new_header = list(datasets_short.values())
    new_header.insert(0, " ")
    res = tabulate(comp_list, tablefmt="latex", headers=new_header)
    file.write(res)


# DELAY TABLES
for i in dirs:
    dir = dirs[i]
    df = pd.read_csv(f"../{dir}/Results.csv")

    file = open(f"Iter/iter-{dir}.txt", "w")
    comp_list = []
    for i in iter_column:
        c = iter_column[i]
        temp = []
        for v in datasets.values():
            res = df[df['Problem'].str.contains(v)]
            temp.append(float(res[c]))
        comp_list.append((iter_column_long[i], temp[0], temp[1], temp[2], temp[3], temp[4]))

    res = tabulate(comp_list, tablefmt="latex", headers=datasets_short.values())
    file.write(res)
