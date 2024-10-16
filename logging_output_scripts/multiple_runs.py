import json
from logging_output_scripts.violin_and_swarm_plots import create_plots
from logging_output_scripts.stat_analysis import calvo
from time import sleep
from logging_output_scripts.utils import filter_runs
import mlflow

config_filename = "logging_output_scripts/config.json"

datasets2 = {
        "combined_cycle_power_plant": "Combined Cycle Power Plant",
        "airfoil_self_noise": "Airfoil Self-Noise",
        "concrete_strength": "Concrete Strength",
        "energy_cool": "Energy Efficiency Cooling",
        # "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",   
        # "parkinson_total": "Parkinson's Telemonitoring"
}

datasets = ["concrete_strength", "energy_cool", "combined_cycle_power_plant", "airfoil_self_noise"]
filter_subpop = ["FilterSubpopulation", "NBestFitness", "NRandom", "RouletteWheel"]
experience_calc = ["ExperienceCalculation", "CapExperience", "CapExperienceWithDimensionality"]


with open(config_filename, 'r') as f:
    config = json.load(f)

config["output_directory"] = f"logging_output_scripts/outputs/MIX"
# config["heuristics"] = {"p:energy_cool; r:2; f:FilterSubpopulation; -e:ExperienceCalculation":"2",
#                         "p:energy_cool; r:1; f:FilterSubpopulation; -e:CapExperience":"1",
#                         "p:energy_cool; r:5; f:NBestFitness; -e:CapExperience":"5",
#                         "p:energy_cool; r:1; f:FilterSubpopulation; -e:ExperienceCalculation":"1",
#                         "p:energy_cool; r:3; f:FilterSubpopulation; -e:ExperienceCalculation":"3",
#                         "p:energy_cool; r:2; f:FilterSubpopulation; -e:CapExperience":"2",
#                         }
config["datasets"] = datasets2
config["normalize_datasets"] = True

with open(config_filename, 'w') as f:
    json.dump(config, f)

all_runs_df = mlflow.search_runs(search_all_experiments=True)
filter_runs(all_runs_df)

for dataset in datasets:
    for filter in filter_subpop:
        for experience in experience_calc:
            with open(config_filename, 'r') as f:
                config = json.load(f)

            config["heuristics"] = {f"p:{dataset}; r:{1}; f:{filter}; -e:{experience}.": "1",
                                    f"p:{dataset}; r:{2}; f:{filter}; -e:{experience}.": "2",
                                    f"p:{dataset}; r:{3}; f:{filter}; -e:{experience}.": "3",
                                    f"p:{dataset}; r:{4}; f:{filter}; -e:{experience}.": "4",
                                    f"p:{dataset}; r:{5}; f:{filter}; -e:{experience}.": "5"
                                    }

            config["output_directory"] = f"logging_output_scripts/outputs/mixing/{filter}_{experience}"

            config["datasets"] = {dataset: dataset_mapping[dataset]}

            with open(config_filename, 'w') as f:
                json.dump(config, f)

            sleep(5)

            calvo()

            print("Done:", f"f:{filter}; -e:{experience}")
