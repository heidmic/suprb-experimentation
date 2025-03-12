import json
import decision_tree
from logging_output_scripts.utils import check_and_create_dir
import rf
REGRESSOR_CONFIG_PATH = 'logging_output_scripts/config_regression.json'
CLASSIFIER_CONFIG_PATH = 'logging_output_scripts/config_classification.json'

def run(isClass): 
    config_path = REGRESSOR_CONFIG_PATH
    if isClass:
        config_path = CLASSIFIER_CONFIG_PATH
    with open(config_path) as f:
        config = json.load(f)
    
    final_output_dir = f"{config['output_directory']}"
    
    check_and_create_dir(final_output_dir, "csv_summary")
    # Head of csv-File
    header = "Problem,Mean_depth,Std_depth,Mean_leaves,Std_leaves"
    values = ""
    for problem in config['datasets']:
        # Disable click option in the otherf files to allow calling with params
        #rf.run(problem = problem, job_id = 1)
        #continue
        values += "\n" + problem + ","
        values += decision_tree.run(problem = problem, job_id = 1)
        print(f"Done for {problem} with decision tree")
    with open(f"{final_output_dir}/csv_summary/Tree_comp_summary.csv", "w") as file:
            file.write(header + values) 
     
if __name__ == '__main__':
     run(isClass=True)