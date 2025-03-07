import json
import os
import pandas as pd
import numpy as np
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
json_dir = "output_json\\"

rule_number = 0
input_dir = {0: "Cement [kg/m3]", 1: "Blast Furnace Slag [kg/m3]", 2: "Fly Ash [kg/m3]", 3: "Water [kg/m3]",
             4: "Superplasticizer [kg/m3]", 5: "Coarse Aggregate [kg/m3]", 6: "Fine Aggregate [kg/m3]", 7: "Age [days]"}
original_range = {0: [104.72, 516.78], 1: [0, 359.40], 2: [13.45, 200], 3: [122.64, 244.80], 4: [6.02, 24.80],
                  5: [950.16, 1145], 6: [756.14, 992.60], 7: [18.36, 365]}
input_abalone = {0: "Sex", 1: "Length [mm]", 2: "Diameter [mm]", 3: "Height [mm]", 4: "Whole weight [grams]",
             5: "Shucked weight [grams]", 6: "Viscera weight [grams]", 7: "Shell weight [grams]"}
original_abalone = {0: [0, 2], 1: [0.075, 0.815], 2: [0.055, 0.65], 3: [0.0, 1.13], 4: [0.002, 2.826],
                  5: [0.001, 1.488], 6: [0.001, 0.76], 7: [0.001, 1.005]}

def read_bounds_and_coef(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    bounds = data['pool'][rule_number]['match']['bounds']
    coef = data['pool'][rule_number]['model']['coef_']
    
    return bounds, coef

def read_intercept_and_experience(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    intercept = data['pool'][rule_number]['model']['intercept_']
    experience = data['pool'][rule_number]['experience_']
    return intercept, experience

def make_rule_table(file_name):
    bounds, coef = read_bounds_and_coef(json_dir + file_name + ".json")
    
    # Ensure values are numeric
    coef = np.array(coef, dtype=float)
    #intercept = float(intercept)
    
    # Convert bounds to numpy array of floats
    bounds = np.array([np.array(b, dtype=float) for b in bounds])
    
    # Round values
    coef = np.round(coef, 2)
    bounds = np.round(bounds, 2)

    table = [input_dir.values(), original_range.values(), bounds, coef]
    df = pd.DataFrame(table).T
    #new_row = pd.DataFrame([experience, "", "", np.round(intercept,4)]).T
    #df = pd.concat([df, new_row], ignore_index=True)
    #new_row = pd.DataFrame(["\hline In-sample MSE 1.5310", "In-sample MSE 0.917", "Experience", experience]).T
    #df = pd.concat([df, new_row], ignore_index=True)
    table = df.to_numpy()
    res = tabulate(table, headers=['Input variable', 'Original range', 'Matching bounds', 'Coef'], tablefmt='latex_raw')
    with open(f'{final_output_dir}/latex_tabular/rule_{file_name}.txt', 'w') as file:
        file.write(res)


def create_folder(folder_name):
    directory = folder_name
    if not os.path.exists(directory):
        os.mkdir(directory)

def non_zero_coef(file_name):
    with open(json_dir + file_name + ".json", 'r') as file:
        data = json.load(file)
    pool = data['pool']
    amount = []
    for model in pool:
        coef = model['model']['coef_']
        amount.append(np.count_nonzero(coef))
    #print(amount)
    return f"{round(np.mean(amount),2)} \pm {round(np.std(amount),2)}"

def bound_space(file_name):
    with open(json_dir + file_name + ".json", 'r') as file:
        data = json.load(file)
    pool = data['pool']
    space = []
    for model in pool:
        bounds = model['match']['bounds']
        bounds = np.array([np.array(b, dtype=float) for b in bounds])
        diff = (bounds[:,1] - bounds[:,0])/2
        space.append(np.prod(diff))
    return f"{round(np.mean(space),4)} \pm {round(np.std(space),4)}"

if __name__ == "__main__":
    check_and_create_dir(final_output_dir, "latex_tabular")
    #make_rule_table("abalonel2")
    print(bound_space("CSridge"))
    print(bound_space("CSelasticNet"))
    print(bound_space("CSlasso"))
