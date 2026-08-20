import json
import os
import sys

import mlflow
import numpy as np
import time

from logging_output_scripts import violin_and_swarm_plots
from logging_output_scripts import moo_plots
from logging_output_scripts.stat_analysis import calvo, ttest, cohens_pairwise_d
from logging_output_scripts.utils import filter_runs

saga_datasets = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    # "energy_cool": "Energy Efficiency Cooling",
    "protein_structure": "Physiochemical Properties of Protein Tertiary Structure",
    "parkinson_total": "Parkinson's Telemonitoring",
}

datasets_no_pppts = {
    "combined_cycle_power_plant": "Combined Cycle Power Plant",
    "airfoil_self_noise": "Airfoil Self-Noise",
    "concrete_strength": "Concrete Strength",
    "parkinson_total": "Parkinson's Telemonitoring",
}


def mlruns_to_csv(datasets, subdir, normalize):
    all_runs_df = mlflow.search_runs(search_all_experiments=True)

    print("Dataset\t\t\tMin MSE\tMax MSE\tMin Complexity\tMax Complexity")
    for dataset in datasets:
        mse = "metrics.test_neg_mean_squared_error"
        complexity = "metrics.elitist_complexity"
        hypervolume = "metrics.hypervolume"
        sc_iters = "metrics.sc_iterations"
        spread = "metrics.spread"
        test_hypervolume = "metrics.test_hypervolume"
        df = all_runs_df[
            all_runs_df["tags.mlflow.runName"].str.contains(dataset, case=False, na=False) & (all_runs_df["tags.fold"] == "True")
        ]
        df = df[["tags.mlflow.runName", "artifact_uri", mse, complexity, hypervolume, test_hypervolume, spread, sc_iters]]
        print(
            f"{dataset}\t\t\t{np.min(df[mse]):.4f}\t{np.max(df[mse]):.4f}\t{np.min(df[complexity]):.4f}\t" f"{np.max(df[complexity]):.4f}"
        )

        roots = all_runs_df[
            all_runs_df["tags.mlflow.runName"].str.contains(dataset, case=False, na=False) & (all_runs_df["tags.root"] == "True")
        ]
        roots = roots[["tags.mlflow.runName", "artifact_uri", "params.tuned_params"]]

        df[mse] *= -1
        if normalize:
            df[mse] = (df[mse] - np.min(df[mse])) / (np.max(df[mse]) - np.min(df[mse]))
            df[complexity] = (df[complexity] - np.min(df[complexity])) / (np.max(df[complexity]) - np.min(df[complexity]))
        df.to_csv(f"mlruns_csv/{subdir}/{dataset}_all.csv", index=False)
        roots.to_csv(f"mlruns_csv/{subdir}/{dataset}_roots.csv", index=False)


ga_baseline = {
    "Baseline c:ga32": "GA 32",
    "Baseline c:ga64": "GA 64",
}

ga_baseline_more_tuning = {
    "Baseline c:ga32": "GA 32",
    "Baseline c:ga64": "GA 64",
}

moo_baseline = {
    "Baseline nsga2": "NSGA-II",
    "Baseline nsga3": "U-NSGA-III",
    "Baseline spea2": "SPEA2",
}

spea2_only = {"Baseline spea2": "SPEA2"}

test = {"Test nsga3": "U-NSGA-III", "Test spea2": "SPEA2"}

moo_sampler_all = {
    "SampComp spea2 c:beta_equi_untuned": "Equidistant Untuned",
    "SampComp spea2 c:beta_equi_tuned": "Equidistant Tuned",
    "Baseline spea2": r"Projection Untuned",
    "SampComp spea2 c:beta_proj_tuned": "Projection Tuned",
    "SampComp spea2 c:diversity": "Diversity",
}

moo_sampler_equi_proj = {
    "Baseline spea2": r"Projection Untuned",
    "SampComp spea2 c:beta_equi_untuned": "Equidistant Untuned",
}

moo_early_base_comp = {
    "Baseline nsga2": "NSGA-II",
    "Early Stopping nsga2": "NSGA-II HT",
    "Baseline nsga3": "U-NSGA-III",
    "Early Stopping nsga3": "U-NSGA-III HT",
    "Baseline spea2": "SPEA2",
    "Early Stopping spea2": "SPEA2 HT",
}

moo_early_no_base = {
    "Early Stopping nsga2": "NSGA-II HT",
    "Early Stopping nsga3": "U-NSGA-III HT",
    "Early Stopping spea2": "SPEA2 HT",
}

moo_early_nsga2_spea2 = {
    "Early Stopping spea2": "NSGA-II HT",
    "Early Stopping spea2": "SPEA2 HT",
}

moo_early_only_spea2 = {
    "Early Stopping spea2": "SPEA2 HT",
}

moo_ts_all = {
    "Baseline spea2": "Baseline",
    "TScomp spea2 c:ga-moo e:False": "Staged Naive",
    "TScomp spea2 c:ga-moo e:True": "Staged HT",
    # "TScomp spea2 c:ga_without_tuning-moo e:True": "GA Untuned - MOO HT",
    "TScomp spea2 c:ga-moo_cold_staging e:True": "Staged HT CS",
}

moo_ts_naive = {
    "Baseline spea2": "MOO Baseline",
    "TScomp spea2 c:ga-moo e:False": "GA - MOO",
}

pop_size = {"Baseline spea2": "$N = 32$", "PopComp spea2 c:64": "$N = 64$", "PopComp spea2 c:128": "$N = 128$"}

more_rules = {"Baseline nsga2": "128 Rules", "MoreRules nsga2": "1024 Rules"}


def run_main():
    with open("logging_output_scripts/config.json", "r") as f:
        config = json.load(f)

    config["datasets"] = saga_datasets if setting is not test else {"parkinson_total": "Parkinson's Telemonitoring"}

    config["output_directory"] = setting[0]
    if not os.path.isdir("diss-graphs/graphs"):
        os.mkdir("diss-graphs/graphs")

    if not os.path.isdir(config["output_directory"]):
        os.mkdir(config["output_directory"])

    config["normalize_datasets"] = setting[3]

    config["heuristics"] = setting[1]
    config["reference_heuristics"] = setting[5] if len(setting) > 5 else {}
    config["data_directory"] = setting[4]

    with open("logging_output_scripts/config.json", "w") as f:
        json.dump(config, f)

    time.sleep(10)

    if config["data_directory"] == "mlruns":
        all_runs_df = mlflow.search_runs(search_all_experiments=True)
        filter_runs(all_runs_df)

    if setting[0] == "diss-graphs/graphs/MOO-baseline":
        cohens_pairwise_d(
            [("Baseline nsga2", "Baseline spea2"), ("Baseline nsga2", "Baseline nsga3"), ("Baseline nsga3", "Baseline spea2")],
            ["NSGA-II - SPEA2", "NSGA-II - U-NSGA-III", "U-NSGA-III - SPEA2"],
        )

        ttest(latex=True, cand1="Baseline nsga2", cand2="Baseline spea2", cand1_name="NSGA-II", cand2_name="SPEA2")
        ttest(latex=True, cand1="Baseline nsga2", cand2="Baseline nsga3", cand1_name="NSGA-II", cand2_name="U-NSGA-III")

        ttest(latex=True, cand1="Baseline nsga3", cand2="Baseline spea2", cand1_name="U-NSGA-III", cand2_name="SPEA2")

    if setting[0] == "diss-graphs/graphs/HT-comparison":
        cohens_pairwise_d(
            [
                ("Baseline nsga2", "Early Stopping nsga2"),
                ("Baseline spea2", "Early Stopping spea2"),
                ("Early Stopping nsga2", "Early Stopping spea2"),
            ],
            ["NSGA-II - NSGA-II HT", "SPEA2 - SPEA2 HT", "NSGA-II HT - SPEA2 HT"],
        )
        ttest(latex=True, cand1="Baseline nsga2", cand2="Early Stopping nsga2", cand1_name="NSGA-II", cand2_name="NSGA-II HT")
        ttest(latex=True, cand1="Baseline nsga3", cand2="Early Stopping nsga3", cand1_name="U-NSGA-III", cand2_name="U-NSGA-III HT")
        ttest(latex=True, cand1="Baseline spea2", cand2="Early Stopping spea2", cand1_name="SPEA2", cand2_name="SPEA2 HT")

        ttest(latex=True, cand1="Early Stopping nsga2", cand2="Early Stopping spea2", cand1_name="NSGA-II HT", cand2_name="SPEA2 HT")

    if setting[0] == "diss-graphs/graphs/SAMPLER_ALL":
        ttest(
            latex=True,
            cand1="SampComp spea2 c:beta_equi_untuned",
            cand2="Baseline spea2",
            cand1_name="Equidistant Untuned",
            cand2_name="Projection Untuned",
        )
        ttest(
            latex=True,
            cand1="Baseline spea2",
            cand2="SampComp spea2 c:beta_proj_tuned",
            cand1_name="Projection Untuned",
            cand2_name="Projection Tuned",
        )
        ttest(
            latex=True,
            cand1="SampComp spea2 c:beta_equi_untuned",
            cand2="SampComp spea2 c:beta_equi_tuned",
            cand1_name="Equidistant Untuned",
            cand2_name="Equidistant Tuned",
        )

        ttest(
            latex=True, cand1="Baseline spea2", cand2="SampComp spea2 c:diversity", cand1_name="Projection Untuned", cand2_name="Diversity"
        )

    if setting[0] == "diss-graphs/graphs/TS_ALL":
        ttest(latex=True, cand1="Baseline spea2", cand2="TScomp spea2 c:ga-moo e:False", cand1_name="MOO Baseline", cand2_name="GA - MOO")
        ttest(
            latex=True,
            cand1="TScomp spea2 c:ga-moo e:True",
            cand2="TScomp spea2 c:ga-moo_cold_staging e:True",
            cand1_name="Staged HT",
            cand2_name="Staged HT CS",
        )
        ttest(
            latex=True,
            cand1="Baseline spea2",
            cand2="TScomp spea2 c:ga-moo_cold_staging e:True",
            cand1_name="MOO Baseline",
            cand2_name="Staged HT CS",
        )

    if setting[0] == "diss-graphs/graphs/POP":
        ttest(latex=True, cand1="Baseline spea2", cand2="PopComp spea2 c:128", cand1_name="$N = 32$", cand2_name="$N = 128$")

    if len(config["heuristics"]) > 1:
        calvo(ylabel=setting[2])

    moo_plots.create_plots()

    # violin_and_swarm_plots.create_plots()


if __name__ == "__main__":
    ga_base = ["diss-graphs/graphs/GA_BASELINE", ga_baseline, "Solution Composition", False, "mlruns_csv/GA_BASELINE"]
    moo_algos = ["diss-graphs/graphs/MOO-baseline", moo_baseline, "Configuration", False, "mlruns_csv/MOO", ga_baseline_more_tuning]
    moo_sampler_all = ["diss-graphs/graphs/SAMPLER_ALL", moo_sampler_all, "Configuration", False, "mlruns_csv/SAMPLER_ALL"]
    moo_sampler_equi_proj = [
        "diss-graphs/graphs/SAMPLER_EQUI_PROJ",
        moo_sampler_equi_proj,
        "Configuration",
        False,
        "mlruns_csv/SAMPLER_EQUI_PROJ",
    ]
    moo_early_base_comp = [
        "diss-graphs/graphs/HT-comparison",
        moo_early_base_comp,
        "Configuration",
        False,
        "mlruns_csv/EARLY",
        ga_baseline_more_tuning,
    ]
    moo_early_no_base = [
        "diss-graphs/graphs/HT-only",
        moo_early_no_base,
        "Configuration",
        False,
        "mlruns_csv/EARLY_NO_BASE",
        ga_baseline_more_tuning,
    ]
    moo_early_nsga2_spea2 = [
        "diss-graphs/graphs/EARLY_NO_BASE_NSGA2_SPEA2",
        moo_early_nsga2_spea2,
        "Configuration",
        False,
        "mlruns_csv/EARLY_NO_BASE_NSGA2_SPEA2",
        ga_baseline_more_tuning,
    ]
    moo_early_only_spea2 = [
        "diss-graphs/graphs/EARLY_NO_BASE_SPEA2",
        moo_early_only_spea2,
        "Configuration",
        False,
        "mlruns_csv/EARLY_NO_BASE_SPEA2",
        ga_baseline_more_tuning,
    ]
    moo_ts_all = ["diss-graphs/graphs/TS_ALL", moo_ts_all, "Configuration", False, "mlruns_csv/TS_ALL"]
    moo_ts_naive = ["diss-graphs/graphs/TS_NAIVE", moo_ts_naive, "Configuration", False, "mlruns_csv/TS_NAIVE"]
    pop_size = ["diss-graphs/graphs/POP", pop_size, "Configuration", False, "mlruns_csv/POP"]
    more_rules = ["diss-graphs/graphs/MORE_RULES", more_rules, "Configuration", False, "mlruns_csv/MORE_RULES"]
    test = ["diss-graphs/graphs/TEST", test, "Configuration", False, "mlruns_csv/TEST"]
    spea2_only = ["diss-graphs/graphs/SPEA2_ONLY", spea2_only, "Configuration", False, "mlruns_csv/SPEA2_ONLY", ga_baseline_more_tuning]

    # setting = ga_base
    # setting = test
    # setting = moo_algos  # Check
    # setting = moo_sampler_all                 # Check
    # setting = moo_sampler_equi_proj           # Check
    # setting = moo_early_base_comp             # Check
    setting = moo_early_no_base  # Check
    # setting = moo_early_nsga2_spea2
    # setting = moo_early_only_spea2            # Check
    # setting = moo_ts_all                      # Check
    # setting = moo_ts_naive                    # Check
    # setting = pop_size
    # setting = more_rules
    # setting = spea2_only

    mlruns_to_csv(
        saga_datasets if setting is not test else {"parkinson_total": "Parkinson's Telemonitoring"},
        subdir=setting[4].split("/")[-1],
        normalize=True,
    )
    run_main()

    print(f"\nFinished creating plots for {setting[4].split("/")[-1]}")
