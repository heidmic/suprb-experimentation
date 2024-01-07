import json
from logging_output_scripts.violin_and_swarm_plots import create_plots
from logging_output_scripts.stat_analysis import calvo
from time import sleep

config_filename = "config.json"

dataset_mapping = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self Noise",
    "concrete_strength": "Concrete Strength",
    "energy_cool": "Energy Efficiency Cooling"
}

datasets = ["concrete_strength", "energy_cool", "combined_cycle_power_plant", "airfoil_self_noise"]
filter_subpop = ["FilterSubpopulation", "NBestFitness", "NRandom", "RouletteWheel"]
experience_calc = ["ExperienceCalculation", "CapExperience", "CapExperienceWithDimensionality"]

for dataset in datasets:
    for filter in filter_subpop:
        for experience in experience_calc:
            with open(config_filename, 'r') as f:
                config = json.load(f)

            config["heuristics"] = {f"p:{dataset}; r:{1}; f:{filter}; -e:{experience}.": "tuning ASN",
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