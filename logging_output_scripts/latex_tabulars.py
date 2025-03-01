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
datasets_short = {0: 'CS', 1: 'ASN' , 2: 'CCPP', 3: 'EH'}

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


def write_complexity_all():
    comp_list = []
    for model in config["model_names"]:
        df = pd.read_csv(f"{summary_csv_dir}/{config['model_names'][model]}_summary.csv")
        for column_name, column_name_short in zip(comp_column.values(), comp_column_short.values()):
            data_res = load_problem_columns(df, column_name)
            comp_list.append((config["model_names"][model], column_name_short, data_res[0], data_res[1], data_res[2], data_res[3]))

    datasets = [*config["datasets"].values()]
    res = tabulate(comp_list, tablefmt="latex_booktabs", headers=['Model', 'Metric', datasets[0], datasets[1], datasets[2], datasets[3]])

    with open(f"{final_output_dir}/latex_tabular/ComplexityAll.txt", "w") as file:
        file.write(res)

# COMPLEXITY TABLES


def write_complexity():
    for model in config["model_names"]:
        df = pd.read_csv(f"{summary_csv_dir}/{config['model_names'][model]}_summary.csv")
        comp_list = []
        for column_name, column_name_short in zip(comp_column.values(), comp_column_short.values()):
            data_res = load_problem_columns(df, column_name)
            comp_list.append((column_name_short, data_res[0], data_res[1], data_res[2], data_res[3]))

        datasets = [*config["datasets"].values()]
        res = tabulate(comp_list, tablefmt="latex_booktabs", headers=['Metric', datasets[0], datasets[1], datasets[2], datasets[3]])

        with open(f"{final_output_dir}/latex_tabular/Complexity-{model}.txt", "w") as file:
            file.write(res)

# GENOMES (Requires Genomes to be Stored as a .json in the respective Folder) [Leave out if not needed]


def write_genomes():
    for model in config["model_names"]:
        for problem in config["datasets"]:
            with open(f'Genomes/{model}_{problem}.json', 'r') as f:
                genomes = json.load(f)

            genomes = list(genomes.values())

            genome_list = []
            # Create tuple of shape (iteration, genome[iteration])
            for x in range(len(genomes)):
                genome_list.append((str(x), genomes[x]))

            headers = ['Iteration', 'Genome']
            genome_table = tabulate(genome_list, tablefmt="latex_booktabs", headers=headers)

            create_folder("Genomes")

            with open(f"Genomes/{model}-{problem}.txt", "w") as file:
                file.write(genome_table)


def write_error_all(datasets_short):
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
        for model in config["model_names"]:
            df = pd.read_csv(f"{summary_csv_dir}/{config['model_names'][model]}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(((round(float(res['MEAN_ERROR']), len(config["datasets"]))), round(float(res['STD_ERROR']), 5)))
        column.append(row)

    res = tabulate(column, tablefmt="latex_booktabs", headers=datasets_short.values())

    with open(f"{final_output_dir}/latex_tabular/ERRORAll.txt", "w") as file:
        file.write(res)


# ERROR TABLES (Creates individual tables to be combined into a larger tabular)
def write_error():
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
        for model, renamed_model in config['model_names'].items():
            df = pd.read_csv(f"{summary_csv_dir}/{renamed_model}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(((round(float(res['MEAN_ERROR']), 4)), round(float(res['STD_ERROR']), 4)))
        column.append(row)

    headers = ['ERROR', 'STD']
    problem_1 = tabulate(column[0], tablefmt="latex_booktabs", headers=headers)
    problem_2 = tabulate(column[1], tablefmt="latex_booktabs", headers=headers)
    problem_3 = tabulate(column[2], tablefmt="latex_booktabs", headers=headers)
    problem_4 = tabulate(column[3], tablefmt="latex_booktabs", headers=headers)

    with open(f"{final_output_dir}/latex_tabular/ERROR.txt", "w") as file:
        file.write(problem_1 + "\n\n" + problem_2 + "\n\n" + problem_3 + "\n\n" + problem_4)


def single_table(dataset_shorts):
    columns = []
    for problem in config["datasets"]:
        # Each row features one problem for one model
        row = []
        row.append(config["datasets"][problem])
        for model in config["model_names"]:
            df = pd.read_csv(f"{summary_csv_dir}/{config['model_names'][model]}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(str(round(float(res['MEAN_ERROR']), 2))+"pm" +
                       str(round(float(res['STD_ERROR']), 2)))
            row.append(str(round(float(res['MEAN_COMP']), 2))+"pm" +
                       str(round(float(res['STD_COMP']), 2)))
        columns.append(row)
    frame = pd.DataFrame(columns)
    headers = [x for y in [['Error', 'Complexity'] for i in range(
        frame.shape[1]-1)] for x in y]
    latex = tabulate(columns, tablefmt="latex_booktabs", headers=headers)
    splits = latex.split("\\toprule")
    model_names = [*config["model_names"].values()]
    headline = " ".join(["\\multicolumn{2}{c}{"+h+"} &" for h in model_names])
    latex = splits[0]+"\\toprule"+headline+splits[1]
    with open(f"{final_output_dir}/latex_tabular/combined.txt", "w") as file:
        file.write(latex)


def single_table_all_error(dataset_shorts):
    columns = []
    for model in config["model_names"]:
        # Each row features one problem for one model
        row = [utils.datasets_map[model]]
        for problem in config["datasets"]:
            df = pd.read_csv(f"{summary_csv_dir}/{config['model_names'][model]}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(str(round(float(res['MEAN_ERROR']), 2))+"pm" +
                       str(round(float(res['STD_ERROR']), 2)))
        columns.append(row)
    latex = tabulate(columns, tablefmt="latex_booktabs")
    splits = latex.split("\\toprule")
    headline = " &".join(dataset_shorts.values())
    headline =  "\n &" + headline + "\\\\"
    latex = splits[0]+"\\toprule"+headline+splits[1]
    with open(f"{final_output_dir}/latex_tabular/latex_error.txt", "w") as file:
        file.write(latex)


def single_table_all_complexity(dataset_shorts):
    columns = []
    for model in config["model_names"]:
        # Each row features one problem for one model
        row = [utils.datasets_map[model]]
        for problem in config["datasets"]:
            df = pd.read_csv(f"{summary_csv_dir}/{config['model_names'][model]}_summary.csv")
            res = df[df['Problem'].str.contains(problem)]
            row.append(str(round(float(res['MEAN_COMP']), 2))+"pm" +
                       str(round(float(res['STD_COMP']), 2)))
        columns.append(row)
    latex = tabulate(columns, tablefmt="latex_booktabs")
    splits = latex.split("\\toprule")
    headline = " &".join(dataset_shorts.values())
    headline =  "\n &" + headline + "\\\\"
    latex = splits[0]+"\\toprule"+headline+splits[1]
    with open(f"{final_output_dir}/latex_tabular/latex_complexity.txt", "w") as file:
        file.write(latex)

def swaps_error(dataset_shorts, base_model):
    columns = []
    df = pd.read_csv(f"{summary_csv_dir}/{config['model_names'][base_model]}_swaps_summary.csv")
    for model in config["model_names"]:
        # Each row features one problem for one model
        row = [utils.datasets_map[model]]
        for problem in config["datasets"]:
            res = df[df['Problem'].str.contains(problem + " " + f"n:{model}")]
            row.append(str(round(float(res['MEAN_ERROR']), 2))+"pm" +
                       str(round(float(res['STD_ERROR']), 2)))
        columns.append(row)
    latex = tabulate(columns, tablefmt="latex_booktabs")
    splits = latex.split("\\toprule")
    headline = " &".join(dataset_shorts.values())
    headline =  "\n &" + headline + "\\\\"
    latex = splits[0]+"\\toprule"+headline+splits[1]
    with open(f"{final_output_dir}/latex_tabular/latex_{base_model}_swaps.txt", "w") as file:
        file.write(latex)
        

# Add / leave out certain tables
if __name__ == '__main__':
    final_output_dir = f"{config['output_directory']}"
    check_and_create_dir(final_output_dir, 'latex_tabular')
    # write_complexity()
    # write_complexity_all()
    # Only use if genomes are actually tracked.
    # write_genomes()
    # write_error()
    #write_error_all(datasets_short)
    single_table(dataset_shorts = datasets_short)
    single_table_all_error(dataset_shorts = datasets_short)
    single_table_all_complexity(dataset_shorts=datasets_short)
    for model in config["model_names"]:
        swaps_error(dataset_shorts=datasets_short, base_model=model)
