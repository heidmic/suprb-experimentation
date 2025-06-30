import json
import os
import mlflow
import numpy as np
import time

from logging_output_scripts.violin_and_swarm_plots import create_plots
from logging_output_scripts.summary_csv import create_summary_csv
from logging_output_scripts.stat_analysis import calvo, ttest
from logging_output_scripts.latex_tabulars import write_complexity_all
from logging_output_scripts.utils import filter_runs
from logging_output_scripts import latex_tabulars

datasets = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    "energy_cool": "Energy Efficiency Cooling",
    "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",
    "parkinson_total": "Parkinson's Telemonitoring"
}

saga_datasets = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    # "energy_cool": "Energy Efficiency Cooling",
    "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",
    "parkinson_total": "Parkinson's Telemonitoring"
}

mix_datasets = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    "energy_cool": "Energy Efficiency Cooling"
}

solution_composition = {
    "GeneticAlgorithm": "GA",
    "RandomSearch": "RS",
    "ArtificialBeeColonyAlgorithm": "ABC",
    "AntColonyOptimization": "ACO",
    "GreyWolfOptimizer": "GWO",
    "ParticleSwarmOptimization": "PSO"
}

sc_mix_rd = {
    "ES Tuning": "GA",
    "RandomSearch": "RS",
    "ArtificialBeeColonyAlgorithm": "ABC",
    "AntColonyOptimization": "ACO",
    "GreyWolfOptimizer": "GWO",
    "ParticleSwarmOptimization": "PSO"
}


rule_discovery = {
    "ES Tuning": "ES",
    "RS Tuning": "RS",
    " NS True": "NS-P",
    "MCNS True": "MCNS-P",
    "NSLC True": "NSLC-P",
    " NS False": "NS-G",
    "MCNS False": "MCNS-G",
    "NSLC False": "NSLC-G",
    # "NSLC Tuning": "NSLC-P"
}

asoc = {
    "ES Tuning": "SupRB",
    "XCSF": "XCSF",
    "Decision Tree": "DT",
    "Random Forest": "RF",
}

mixing1 = {"r:1; f:FilterSubpopulation; -e:ExperienceCalculation": "1",
           "r:2; f:FilterSubpopulation; -e:ExperienceCalculation": "2",
           "r:3; f:FilterSubpopulation; -e:ExperienceCalculation": "3",
           "r:4; f:FilterSubpopulation; -e:ExperienceCalculation": "4",
           "r:5; f:FilterSubpopulation; -e:ExperienceCalculation": "5", }

mixing2 = {"r:1; f:FilterSubpopulation; -e:CapExperience/": "1",
           "r:2; f:FilterSubpopulation; -e:CapExperience/": "2",
           "r:3; f:FilterSubpopulation; -e:CapExperience/": "3",
           "r:4; f:FilterSubpopulation; -e:CapExperience/": "4",
           "r:5; f:FilterSubpopulation; -e:CapExperience/": "5", }

mixing3 = {"r:1; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "1",
           "r:2; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "2",
           "r:3; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "3",
           "r:4; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "4",
           "r:5; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "5", }

mixing4 = {"r:1; f:NBestFitness; -e:ExperienceCalculation": "1",
           "r:2; f:NBestFitness; -e:ExperienceCalculation": "2",
           "r:3; f:NBestFitness; -e:ExperienceCalculation": "3",
           "r:4; f:NBestFitness; -e:ExperienceCalculation": "4",
           "r:5; f:NBestFitness; -e:ExperienceCalculation": "5", }

mixing5 = {"r:1; f:NBestFitness; -e:CapExperience/": "1",
           "r:2; f:NBestFitness; -e:CapExperience/": "2",
           "r:3; f:NBestFitness; -e:CapExperience/": "3",
           "r:4; f:NBestFitness; -e:CapExperience/": "4",
           "r:5; f:NBestFitness; -e:CapExperience/": "5", }

mixing6 = {"r:1; f:NBestFitness; -e:CapExperienceWithDimensionality": "1",
           "r:2; f:NBestFitness; -e:CapExperienceWithDimensionality": "2",
           "r:3; f:NBestFitness; -e:CapExperienceWithDimensionality": "3",
           "r:4; f:NBestFitness; -e:CapExperienceWithDimensionality": "4",
           "r:5; f:NBestFitness; -e:CapExperienceWithDimensionality": "5", }

mixing7 = {"r:1; f:NRandom; -e:ExperienceCalculation": "1",
           "r:2; f:NRandom; -e:ExperienceCalculation": "2",
           "r:3; f:NRandom; -e:ExperienceCalculation": "3",
           "r:4; f:NRandom; -e:ExperienceCalculation": "4",
           "r:5; f:NRandom; -e:ExperienceCalculation": "5", }

mixing8 = {"r:1; f:NRandom; -e:CapExperience/": "1",
           "r:2; f:NRandom; -e:CapExperience/": "2",
           "r:3; f:NRandom; -e:CapExperience/": "3",
           "r:4; f:NRandom; -e:CapExperience/": "4",
           "r:5; f:NRandom; -e:CapExperience/": "5", }

mixing9 = {"r:1; f:NRandom; -e:CapExperienceWithDimensionality": "1",
           "r:2; f:NRandom; -e:CapExperienceWithDimensionality": "2",
           "r:3; f:NRandom; -e:CapExperienceWithDimensionality": "3",
           "r:4; f:NRandom; -e:CapExperienceWithDimensionality": "4",
           "r:5; f:NRandom; -e:CapExperienceWithDimensionality": "5", }

mixing10 = {"r:1; f:RouletteWheel; -e:ExperienceCalculation": "1",
            "r:2; f:RouletteWheel; -e:ExperienceCalculation": "2",
            "r:3; f:RouletteWheel; -e:ExperienceCalculation": "3",
            "r:4; f:RouletteWheel; -e:ExperienceCalculation": "4",
            "r:5; f:RouletteWheel; -e:ExperienceCalculation": "5", }

mixing11 = {"r:1; f:RouletteWheel; -e:CapExperience/": "1",
            "r:2; f:RouletteWheel; -e:CapExperience/": "2",
            "r:3; f:RouletteWheel; -e:CapExperience/": "3",
            "r:4; f:RouletteWheel; -e:CapExperience/": "4",
            "r:5; f:RouletteWheel; -e:CapExperience/": "5", }

mixing12 = {"r:1; f:RouletteWheel; -e:CapExperienceWithDimensionality": "1",
            "r:2; f:RouletteWheel; -e:CapExperienceWithDimensionality": "2",
            "r:3; f:RouletteWheel; -e:CapExperienceWithDimensionality": "3",
            "r:4; f:RouletteWheel; -e:CapExperienceWithDimensionality": "4",
            "r:5; f:RouletteWheel; -e:CapExperienceWithDimensionality": "5", }

mixing_calvo = {
    "r:3; f:FilterSubpopulation; -e:ExperienceCalculation": "Base",
    "r:3; f:FilterSubpopulation; -e:CapExperience/": "Experience Cap",
    "r:3; f:FilterSubpopulation; -e:CapExperienceWithDimensionality": "Experience Cap (dim)",
    "r:3; f:NBestFitness; -e:ExperienceCalculation": r"$l$ Best",
    "r:3; f:NBestFitness; -e:CapExperience/": r"$l$ Best & Experience Cap",
    "r:3; f:NBestFitness; -e:CapExperienceWithDimensionality": r"$l$ Best & Experience Cap (dim)",
    "r:3; f:NRandom; -e:ExperienceCalculation": r"$l$ Random",
    "r:3; f:NRandom; -e:CapExperience/": r"$l$ Random & Experience Cap",
    "r:3; f:NRandom; -e:CapExperienceWithDimensionality": r"$l$ Random & Experience Cap (dim)",
    "r:3; f:RouletteWheel; -e:ExperienceCalculation": "RouletteWheel",
    "r:3; f:RouletteWheel; -e:CapExperience/": "RouletteWheel & Experience Cap",
    "r:3; f:RouletteWheel; -e:CapExperienceWithDimensionality": "RouletteWheel & Experience Cap (dim)"
}

mixing_calvo_subset = {
    "r:3; f:NBestFitness; -e:ExperienceCalculation": r"$l$ Best",
    "r:3; f:NBestFitness; -e:CapExperience/": r"$l$ Best & Experience Cap",
    "r:3; f:NBestFitness; -e:CapExperienceWithDimensionality": r"$l$ Best & Experience Cap (dim)",
    "r:3; f:NRandom; -e:ExperienceCalculation": r"$l$ Random",
    "r:3; f:NRandom; -e:CapExperience/": r"$l$ Random & Experience Cap",
    "r:3; f:NRandom; -e:CapExperienceWithDimensionality": r"$l$ Random & Experience Cap (dim)",
    "r:3; f:RouletteWheel; -e:ExperienceCalculation": "RouletteWheel",
    "r:3; f:RouletteWheel; -e:CapExperience/": "RouletteWheel & Experience Cap",
    "r:3; f:RouletteWheel; -e:CapExperienceWithDimensionality": "RouletteWheel & Experience Cap (dim)"
}

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


def mlruns_to_csv(datasets, subdir, normalize):
    all_runs_df = mlflow.search_runs(search_all_experiments=True)

    for dataset in datasets:
        mse = "metrics.test_neg_mean_squared_error"
        complexity = "metrics.elitist_complexity"

        df = all_runs_df[all_runs_df["tags.mlflow.runName"].str.contains(
            dataset, case=False, na=False) & (all_runs_df["tags.fold"] == 'True')]
        df = df[["tags.mlflow.runName", mse, complexity]]
        print(dataset, np.min(df[mse]), np.max(df[mse]), np.min(df[complexity]), np.max(df[complexity]))

        df[mse] *= -1
        if normalize:
            df[mse] = (df[mse] - np.min(df[mse])) / (np.max(df[mse]) - np.min(df[mse]))
            df[complexity] = (df[complexity] - np.min(df[complexity])) / (np.max(df[complexity]) - np.min(df[complexity]))
        df.to_csv(f"mlruns_csv/{subdir}/{dataset}_all.csv", index=False)


saga = {
    "s:ga": "GA",
    "s:saga1": "SAGA1",
    "s:saga2": "SAGA2",
    "s:saga3": "SAGA3",
    "s:sas": "SAGA4"
}

adel = {"SupRB": "SupRB",
        "Random Forest": "RF",
        "Decision Tree": "DT", }


def run_main():
    with open("logging_output_scripts/config.json", "r") as f:
        config = json.load(f)

    # config["datasets"] = {"":""}
    config["datasets"] = datasets
    if setting[0] == "diss-graphs/graphs/SAGA":
        config["datasets"] = saga_datasets
    if setting[0] == "diss-graphs/graphs/MIX" or setting[0] == "diss-graphs/graphs/RBML":
        config["datasets"] = mix_datasets

    config["output_directory"] = setting[0]
    if not os.path.isdir("diss-graphs/graphs"):
        os.mkdir("diss-graphs/graphs")

    if not os.path.isdir(config["output_directory"]):
        os.mkdir(config["output_directory"])

    config["normalize_datasets"] = setting[3]

    config["heuristics"] = setting[1]
    config["data_directory"] = setting[4]

    with open("logging_output_scripts/config.json", "w") as f:
        json.dump(config, f)

    time.sleep(10)

    if config["data_directory"] == "mlruns":
        all_runs_df = mlflow.search_runs(search_all_experiments=True)
        filter_runs(all_runs_df)

    create_plots()
    calvo(ylabel=setting[2])

    if setting[0] == "diss-graphs/graphs/RBML":
        ttest(latex=False, cand1="XCSF", cand2="ES Tuning", cand1_name="XCSF", cand2_name="SupRB")
        ttest(latex=False, cand1="Decision Tree", cand2="ES Tuning", cand1_name="Decision Tree", cand2_name="SupRB")
        ttest(latex=False, cand1="Random Forest", cand2="ES Tuning", cand1_name="Random Forest", cand2_name="SupRB")

    if setting[0] == "diss-graphs/graphs/RD":
        ttest(latex=False, cand1="NSLC True", cand2="ES Tuning", cand1_name="NSLC-P", cand2_name="ES")
        ttest(latex=False, cand1="NSLC False", cand2="ES Tuning", cand1_name="NSLC-G", cand2_name="ES")
        ttest(latex=False, cand1="MCNS True", cand2="ES Tuning", cand1_name="MCNS-P", cand2_name="ES")
        ttest(latex=False, cand1="MCNS False", cand2="ES Tuning", cand1_name="MCNS-G", cand2_name="ES")
        ttest(latex=False, cand1=" NS True", cand2="ES Tuning", cand1_name="NS-P", cand2_name="ES")
        ttest(latex=False, cand1=" NS False", cand2="ES Tuning", cand1_name="NS-G", cand2_name="ES")

        ttest(latex=False, cand1="NSLC False", cand2="NSLC True", cand1_name="NSLC-G", cand2_name="NSLC-P")
        ttest(latex=False, cand1="MCNS True", cand2="NSLC True", cand1_name="MCNS-P", cand2_name="NSLC-P")
        ttest(latex=False, cand1="MCNS False", cand2="NSLC True", cand1_name="MCNS-G", cand2_name="NSLC-P")
        ttest(latex=False, cand1=" NS True", cand2="NSLC True", cand1_name="NS-P", cand2_name="NSLC-P")
        ttest(latex=False, cand1=" NS False", cand2="NSLC True", cand1_name="NS-G", cand2_name="NSLC-P")

        ttest(latex=False, cand1="MCNS True", cand2="NSLC False", cand1_name="MCNS-P", cand2_name="NSLC-G")
        ttest(latex=False, cand1="MCNS False", cand2="NSLC False", cand1_name="MCNS-G", cand2_name="NSLC-G")
        ttest(latex=False, cand1=" NS True", cand2="NSLC False", cand1_name="NS-P", cand2_name="NSLC-G")
        ttest(latex=False, cand1=" NS False", cand2="NSLC False", cand1_name="NS-G", cand2_name="NSLC-G")

        ttest(latex=False, cand1="MCNS False", cand2="MCNS True", cand1_name="MCNS-G", cand2_name="MCNS-P")
        ttest(latex=False, cand1=" NS True", cand2="MCNS True", cand1_name="NS-P", cand2_name="MCNS-P")
        ttest(latex=False, cand1=" NS False", cand2="MCNS True", cand1_name="NS-G", cand2_name="MCNS-P")

        ttest(latex=False, cand1=" NS True", cand2="MCNS False", cand1_name="NS-P", cand2_name="MCNS-G")
        ttest(latex=False, cand1=" NS False", cand2="MCNS False", cand1_name="NS-G", cand2_name="MCNS-G")

        ttest(latex=False, cand1=" NS False", cand2=" NS True", cand1_name="NS-G", cand2_name="NS-P")

    if setting[0] == "diss-graphs/graphs/SC":
        ga_switch = "ES Tuning"  # "GeneticAlgorithm"
        ttest(latex=False, cand1="RandomSearch", cand2=ga_switch, cand1_name="RS", cand2_name="GA")
        ttest(latex=False, cand1="ArtificialBeeColonyAlgorithm", cand2=ga_switch, cand1_name="ABC", cand2_name="GA")
        ttest(latex=False, cand1="AntColonyOptimization", cand2=ga_switch, cand1_name="ACO", cand2_name="GA")
        ttest(latex=False, cand1="GreyWolfOptimizer", cand2=ga_switch, cand1_name="GWO", cand2_name="GA")
        ttest(latex=False, cand1="ParticleSwarmOptimization", cand2=ga_switch, cand1_name="PSO", cand2_name="GA")

    if setting[0] == "diss-graphs/graphs/MIX":
        if setting[4] != "mlruns_csv/MIX/subset_":
            ttest(latex=False, cand1="r:3; f:NBestFitness; -e:ExperienceCalculation",
                  cand2="r:3; f:FilterSubpopulation; -e:ExperienceCalculation", cand1_name=r"$l$ Best", cand2_name="Base")

        ttest(latex=False, cand1="r:3; f:FilterSubpopulation; -e:CapExperience/",
              cand2="r:3; f:FilterSubpopulation; -e:ExperienceCalculation", cand1_name="Experience Cap", cand2_name="Base")
        ttest(latex=False, cand1="r:3; f:FilterSubpopulation; -e:CapExperienceWithDimensionality",
              cand2="r:3; f:FilterSubpopulation; -e:ExperienceCalculation", cand1_name="Experience Cap (dim)", cand2_name="Base")
        ttest(latex=False, cand1="r:3; f:FilterSubpopulation; -e:CapExperience/",
              cand2="r:3; f:FilterSubpopulation; -e:CapExperienceWithDimensionality", cand1_name="Experience Cap", cand2_name="Experience Cap (dim)")

    if setting[0] == "diss-graphs/graphs/SAGA":
        ttest(latex=False, cand1="s:saga1", cand2="s:ga", cand1_name="SAGA1", cand2_name="GA")
        ttest(latex=False, cand1="s:saga2", cand2="s:ga", cand1_name="SAGA2", cand2_name="GA")
        ttest(latex=False, cand1="s:saga3", cand2="s:ga", cand1_name="SAGA3", cand2_name="GA")
        ttest(latex=False, cand1="s:sas", cand2="s:ga", cand1_name="SAGA4", cand2_name="GA")
        ttest(latex=False, cand1="s:saga1", cand2="s:saga2", cand1_name="SAGA1", cand2_name="SAGA2")
        ttest(latex=False, cand1="s:saga1", cand2="s:saga3", cand1_name="SAGA1", cand2_name="SAGA3")
        ttest(latex=False, cand1="s:saga1", cand2="s:sas", cand1_name="SAGA1", cand2_name="SAGA4")
        ttest(latex=False, cand1="s:saga2", cand2="s:saga3", cand1_name="SAGA2", cand2_name="SAGA3")
        ttest(latex=False, cand1="s:saga2", cand2="s:sas", cand1_name="SAGA2", cand2_name="SAGA4")
        ttest(latex=False, cand1="s:saga3", cand2="s:sas", cand1_name="SAGA3", cand2_name="SAGA4")


if __name__ == '__main__':
    rd = ["diss-graphs/graphs/RD", rule_discovery, "Rule Discovery", False, "mlruns_csv/RD"]
    sc = ["diss-graphs/graphs/SC_only_GA", solution_composition, "Solution Composition", False, "mlruns_csv/SC_only_GA"]
    xcsf = ["diss-graphs/graphs/RBML", asoc, "Estimator", False, "mlruns_csv/RBML"]
    adeles = ["diss-graphs/graphs/ADELES", adel, "Rule Discovery", False, "mlruns"]
    mix_calvo = ["diss-graphs/graphs/MIX", mixing_calvo, "Mixing Variant", True, "mlruns_csv/MIX"]
    mix_calvo_sub = ["diss-graphs/graphs/MIX/subset", mixing_calvo_subset, "Mixing Variant", True, "mlruns_csv/MIX"]
    sagas = ["diss-graphs/graphs/SAGA", saga, "Solution Composition", False, "mlruns_csv/SAGA"]
    sc_rd = ["diss-graphs/graphs/SC", sc_mix_rd, "Solution Composition", False, "mlruns_csv/SC"]

    # mlruns_to_csv(datasets, "MIX", True)

    # setting = rd
    # setting = sc
    # setting = sagas
    # setting = mix_calvo
    # setting = mix_calvo_sub
    # setting = xcsf
    setting = sc_rd

    run_main()
    exit()

    for mixing_num in mixing:
        setting = ["diss-graphs/graphs/MIX", mixing_num, "Number of rules participating", True, "mlruns_csv/MIX"]
        run_main()
