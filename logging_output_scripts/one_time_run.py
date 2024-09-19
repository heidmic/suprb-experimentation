import json
import os
import mlflow

from logging_output_scripts.violin_and_swarm_plots import create_plots
from logging_output_scripts.summary_csv import create_summary_csv
from logging_output_scripts.stat_analysis import calvo, ttest
from logging_output_scripts.latex_tabulars import write_complexity_all
from logging_output_scripts.utils import filter_runs

datasets = {
        "combined_cycle_power_plant": "Combined Cycle Power Plant",
        "airfoil_self_noise": "Airfoil Self-Noise",
        "concrete_strength": "Concrete Strength",
        "energy_cool": "Energy Efficiency Cooling",
        # "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",   
        # "parkinson_total": "Parkinson's Telemonitoring"
}

solution_composition = {
        "GeneticAlgorithm": "GA",
        "RandomSearch": "RS",
        "ArtificialBeeColonyAlgorithm": "ABC",
        "AntColonyOptimization": "ACO",
        "GreyWolfOptimizer": "GWO",
        "ParticleSwarmOptimization": "PSW"
    }

rule_discovery = {
        "ES Tuning": "ES",
        "RS Tuning": "RS",
        " NS False": "NS-G",
        "MCNS False": "MCNS-G",
        "NSLC False": "NSLC-G",
        " NS True": "NS-P",
        "MCNS True": "MCNS-P",
        "NSLC True": "NSLC-P",
        # "NSLC Tuning": "NSLC-P"
    }

asoc = {
        "ES Tuning": "ES",
        "XCSF": "XCSF",
        "Random Forest": "RF",
        "Decision Tree": "DT",
}

mixing1 = {"r:1; f:FilterSubpopulation; -e:ExperienceCalculation": "1",
           "r:2; f:FilterSubpopulation; -e:ExperienceCalculation": "2",
           "r:3; f:FilterSubpopulation; -e:ExperienceCalculation": "3",
           "r:4; f:FilterSubpopulation; -e:ExperienceCalculation": "4",
           "r:5; f:FilterSubpopulation; -e:ExperienceCalculation": "5",}

mixing2 = {"r:1; f:FilterSubpopulation; -e:CapExperience/": "1",
           "r:2; f:FilterSubpopulation; -e:CapExperience/": "2",
           "r:3; f:FilterSubpopulation; -e:CapExperience/": "3",
           "r:4; f:FilterSubpopulation; -e:CapExperience/": "4",
           "r:5; f:FilterSubpopulation; -e:CapExperience/": "5",}

mixing3 = {"r:1; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "1",
           "r:2; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "2",
           "r:3; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "3",
           "r:4; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "4",
           "r:5; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "5",}

mixing4 = {"r:1; f:NBestFitness; -e:ExperienceCalculation": "1",
           "r:2; f:NBestFitness; -e:ExperienceCalculation": "2",
           "r:3; f:NBestFitness; -e:ExperienceCalculation": "3",
           "r:4; f:NBestFitness; -e:ExperienceCalculation": "4",
           "r:5; f:NBestFitness; -e:ExperienceCalculation": "5",}

mixing5 = {"r:1; f:NBestFitness; -e:CapExperience/": "1",
           "r:2; f:NBestFitness; -e:CapExperience/": "2",
           "r:3; f:NBestFitness; -e:CapExperience/": "3",
           "r:4; f:NBestFitness; -e:CapExperience/": "4",
           "r:5; f:NBestFitness; -e:CapExperience/": "5",}

mixing6 = {"r:1; f:NBestFitness; -e:CapExperienceWithDimensionality": "1",
           "r:2; f:NBestFitness; -e:CapExperienceWithDimensionality": "2",
           "r:3; f:NBestFitness; -e:CapExperienceWithDimensionality": "3",
           "r:4; f:NBestFitness; -e:CapExperienceWithDimensionality": "4",
           "r:5; f:NBestFitness; -e:CapExperienceWithDimensionality": "5",}

mixing7 = {"r:1; f:NRandom; -e:ExperienceCalculation": "1",
           "r:2; f:NRandom; -e:ExperienceCalculation": "2",
           "r:3; f:NRandom; -e:ExperienceCalculation": "3",
           "r:4; f:NRandom; -e:ExperienceCalculation": "4",
           "r:5; f:NRandom; -e:ExperienceCalculation": "5",}

mixing8 = {"r:1; f:NRandom; -e:CapExperience/": "1",
           "r:2; f:NRandom; -e:CapExperience/": "2",
           "r:3; f:NRandom; -e:CapExperience/": "3",
           "r:4; f:NRandom; -e:CapExperience/": "4",
           "r:5; f:NRandom; -e:CapExperience/": "5",}

mixing9 = {"r:1; f:NRandom; -e:CapExperienceWithDimensionality": "1",
           "r:2; f:NRandom; -e:CapExperienceWithDimensionality": "2",
           "r:3; f:NRandom; -e:CapExperienceWithDimensionality": "3",
           "r:4; f:NRandom; -e:CapExperienceWithDimensionality": "4",
           "r:5; f:NRandom; -e:CapExperienceWithDimensionality": "5",}

mixing10 = {"r:1; f:RouletteWheel; -e:ExperienceCalculation": "1",
            "r:2; f:RouletteWheel; -e:ExperienceCalculation": "2",
            "r:3; f:RouletteWheel; -e:ExperienceCalculation": "3",
            "r:4; f:RouletteWheel; -e:ExperienceCalculation": "4",
            "r:5; f:RouletteWheel; -e:ExperienceCalculation": "5",}

mixing11 = {"r:1; f:RouletteWheel; -e:CapExperience/": "1",
            "r:2; f:RouletteWheel; -e:CapExperience/": "2",
            "r:3; f:RouletteWheel; -e:CapExperience/": "3",
            "r:4; f:RouletteWheel; -e:CapExperience/": "4",
            "r:5; f:RouletteWheel; -e:CapExperience/": "5",}

mixing12 = {"r:1; f:RouletteWheel; -e:CapExperienceWithDimensionality": "1",
            "r:2; f:RouletteWheel; -e:CapExperienceWithDimensionality": "2",
            "r:3; f:RouletteWheel; -e:CapExperienceWithDimensionality": "3",
            "r:4; f:RouletteWheel; -e:CapExperienceWithDimensionality": "4",
            "r:5; f:RouletteWheel; -e:CapExperienceWithDimensionality": "5",}

mixing = []
mixing.append(mixing1)
mixing.append(mixing2)
mixing.append(mixing3)
mixing.append(mixing4)
mixing.append(mixing5)
mixing.append(mixing6)
mixing.append(mixing7)
mixing.append(mixing8)
mixing.append(mixing9)
mixing.append(mixing10)
mixing.append(mixing11)
mixing.append(mixing12)


saga = {
        "s:ga": "GA",
        "s:saga1": "SAGA1",
        "s:saga2": "SAGA2",
        "s:saga3": "SAGA3",
        "s:sas": "SAGA4"
    }

adel = {"SupRB":"SupRB",
        "Random Forest": "RF",
        "Decision Tree": "DT",}

if __name__ == '__main__':
    with open("logging_output_scripts/config.json", "r") as f:
        config = json.load(f)

    def run_main():
        config["datasets"] = {"":""}
        # config["datasets"] = datasets
        config["output_directory"] = setting[0]
        if not os.path.isdir("logging_output_scripts/outputs"):
            os.mkdir("logging_output_scripts/outputs")

        if not os.path.isdir(config["output_directory"]):
            os.mkdir(config["output_directory"])

        config["normalize_datasets"] = setting[3]

        config["heuristics"] = setting[1]

        with open("logging_output_scripts/config.json", "w") as f:
            json.dump(config, f)

        filter_runs(all_runs_df)
        create_plots()
        # create_summary_csv()
        # write_complexity_all({-1: ' ', 0: 'CS', 1: 'CCPP', 2: 'ASN', 3: 'PPPTS', 4:'PT'})
        # calvo(ylabel=setting[2])

        # ttest(latex=False, cand1="s:ga", cand2="s:saga2", cand1_name="GA", cand2_name="SAGA2")
        
    rd = ["logging_output_scripts/outputs/RD", rule_discovery, "Rule Discovery", False]
    sc = ["logging_output_scripts/outputs/SC", solution_composition, "Solution Composition", False]
    xcsf = ["logging_output_scripts/outputs/RBML", asoc, "Rule Discovery", False]
    adeles = ["logging_output_scripts/outputs/ADELES", adel, "Rule Discovery", False]

    all_runs_df = mlflow.search_runs(search_all_experiments=True)

    setting = adeles
    run_main()
    exit()

    for mixing_num in mixing:
        setting = ["logging_output_scripts/outputs/MIX", mixing_num, "Number of rules participating", True]
        run_main()