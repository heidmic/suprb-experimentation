import json
import time
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
        "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",   
        "parkinson_total": "Parkinson's Telemonitoring"
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

saga = {
        "s:ga": "GA",
        "s:saga1": "SAGA1",
        "s:saga2": "SAGA2",
        "s:saga3": "SAGA3",
        "s:sas": "SAGA4"
    }


if __name__ == '__main__':
    with open("logging_output_scripts/config.json", "r") as f:
        config = json.load(f)

    config["datasets"] = datasets

    # config["heuristics"] = {0: 0}
    config["heuristics"] = rule_discovery
    # config["heuristics"] = solution_composition
    # config["heuristics"] = saga

    with open("logging_output_scripts/config.json", "w") as f:
        json.dump(config, f)

    filter_runs()
    create_plots()
    # create_summary_csv()
    # write_complexity_all({-1: ' ', 0: 'CS', 1: 'CCPP', 2: 'ASN', 3: 'PPPTS', 4:'PT'})
    # calvo(ylabel="Solution Composition")
    # calvo(ylabel="Rule Discovery")
    # ttest(latex=False, cand1="s:ga", cand2="s:saga2", cand1_name="GA", cand2_name="SAGA2")
    # ttest(latex=False, cand1="s:ga", cand2="s:saga3", cand1_name="GA", cand2_name="SAGA3")
    # ttest(latex=False, cand1="s:ga", cand2="s:sas", cand1_name="GA", cand2_name="SAGA4")
    # ttest(latex=False, cand1="s:saga2", cand2="s:saga3", cand1_name="SAGA2", cand2_name="SAGA3")
    
