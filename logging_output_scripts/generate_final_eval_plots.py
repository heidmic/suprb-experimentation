import contextlib
import io
import json
import os
import sys

import mlflow
import numpy as np
import time

import violin_and_swarm_plots
import moo_plots
import stat_analysis
from stat_analysis import calvo, ttest, cohens_pairwise_d
from utils import filter_runs
from condition_csv_loader import get_csv_df_for_condition

stat_analysis.get_csv_df = get_csv_df_for_condition

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


def _read_random_state(artifact_uri: str):

    if "suprb-experimentation/" in artifact_uri:
        rel = artifact_uri.split("suprb-experimentation/")[-1]
    else:
        rel = artifact_uri.replace("file://", "")
    params_path = os.path.join(rel, "params.json")
    if not os.path.exists(params_path):
        return None
    try:
        with open(params_path) as f:
            return json.load(f).get("random_state")
    except Exception:
        return None


def mlruns_to_csv(datasets, subdir, normalize):
    all_runs_df = mlflow.search_runs(search_all_experiments=True)

    experiments = mlflow.search_experiments()

    experiment_names = {exp.experiment_id: exp.name for exp in experiments}

    all_runs_df["experiment_name"] = all_runs_df["experiment_id"].map(experiment_names)

    extra_cols = [c for c in ["tags.condition", "tags.rd_method", "tags.pruning", "tags.adaptive"] if c in all_runs_df.columns]

    print("Dataset\t\t\tMin MSE\tMax MSE\tMin Complexity\tMax Complexity")
    for dataset in datasets:
        mse = "metrics.test_neg_mean_squared_error"
        complexity = "metrics.elitist_complexity"
        hypervolume = "metrics.hypervolume"
        sc_iters = "metrics.sc_iterations"
        spread = "metrics.spread"
        test_hypervolume = "metrics.test_hypervolume"
        df = all_runs_df[
            all_runs_df["experiment_name"].str.contains(
                f"p:{dataset}",
                case=False,
                na=False,
            )
            & (all_runs_df["tags.fold"] == "True")
        ]
        df = df[
            ["tags.mlflow.runName", "artifact_uri", "run_id", mse, complexity, hypervolume, test_hypervolume, spread, sc_iters] + extra_cols
        ].copy()
        df["params.random_state"] = df["artifact_uri"].apply(_read_random_state)
        n_missing_seed = df["params.random_state"].isna().sum()
        if n_missing_seed > 0:
            print(
                f"  WARNING: {n_missing_seed} row(s) for {dataset} have no readable random_state "
                f"(params.json missing or unreadable at that path) -- these rows will fail seed pairing."
            )
        print(
            f"{dataset}\t\t\t{np.min(df[mse]):.4f}\t{np.max(df[mse]):.4f}\t{np.min(df[complexity]):.4f}\t" f"{np.max(df[complexity]):.4f}"
        )

        roots = all_runs_df[
            all_runs_df["tags.mlflow.runName"].str.contains(dataset, case=False, na=False) & (all_runs_df["tags.root"] == "True")
        ]
        roots = roots[["tags.mlflow.runName", "artifact_uri", "params.tuned_params"] + extra_cols]

        df[mse] *= -1
        if normalize:
            df[mse] = (df[mse] - np.min(df[mse])) / (np.max(df[mse]) - np.min(df[mse]))
            df[complexity] = (df[complexity] - np.min(df[complexity])) / (np.max(df[complexity]) - np.min(df[complexity]))
        os.makedirs(f"mlruns_csv/{subdir}", exist_ok=True)
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

final_eval_es = {
    "es_nopruning_noadaptive": "ES Baseline",
    "es_pruning_noadaptive": "ES + Pruning",
    "es_nopruning_adaptive": "ES + Adaptive",
    "es_pruning_adaptive": "ES + Pruning + Adaptive",
}
final_eval_ns = {
    "ns_nopruning_noadaptive": "NS Baseline",
    "ns_pruning_noadaptive": "NS + Pruning",
    "ns_nopruning_adaptive": "NS + Adaptive",
    "ns_pruning_adaptive": "NS + Pruning + Adaptive",
}


def run_main():
    with open("logging_output_scripts/config.json", "r") as f:
        config = json.load(f)

    config["datasets"] = current_dataset

    config["output_directory"] = setting[0]
    if not os.path.isdir("diss-graphs/graphs"):
        os.makedirs("diss-graphs/graphs")

    if not os.path.isdir(config["output_directory"]):
        os.makedirs(config["output_directory"])

    config["normalize_datasets"] = setting[3]

    config["heuristics"] = setting[1]
    config["reference_heuristics"] = setting[5] if len(setting) > 5 else {}
    config["data_directory"] = setting[4]
    config["use_condition_tags"] = setting[6] if len(setting) > 6 else False

    with open("logging_output_scripts/config.json", "w") as f:
        json.dump(config, f)

    time.sleep(10)

    all_runs_df = mlflow.search_runs(search_all_experiments=True)
    if not config.get("use_condition_tags"):
        filter_runs(all_runs_df)

    if len(config["heuristics"]) > 1:
        try:
            calvo(ylabel=setting[2])
        except Exception as e:
            print(f"WARNING: calvo() failed ({type(e).__name__}: {e}) -- continuing without it.")

    moo_plots.create_plots()

    if config.get("use_condition_tags"):
        rd_method = list(config["heuristics"].keys())[0].split("_")[0]  # "es" or "ns"
        base = f"{rd_method}_nopruning_noadaptive"
        pruning = f"{rd_method}_pruning_noadaptive"
        adaptive = f"{rd_method}_nopruning_adaptive"
        both = f"{rd_method}_pruning_adaptive"
        pairs = [
            (base, pruning, "Baseline", "Pruning"),
            (base, adaptive, "Baseline", "Adaptive"),
            (base, both, "Baseline", "PruningAdaptive"),
            (pruning, both, "Pruning", "PruningAdaptive"),
            (adaptive, both, "Adaptive", "PruningAdaptive"),
            (pruning, adaptive, "Pruning", "Adaptive"),
        ]

        try:
            cohens_pairwise_d(
                [(p[0], p[1]) for p in pairs],
                [f"{p[2]} - {p[3]}" for p in pairs],
            )
        except Exception as e:
            print(f"WARNING: cohens_pairwise_d() failed ({type(e).__name__}: {e}) -- continuing without it.")

        summaries_dir = os.path.join(config["output_directory"], "tables", "ttest_summaries")
        os.makedirs(summaries_dir, exist_ok=True)
        for cand1, cand2, name1, name2 in pairs:
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    ttest(latex=True, cand1=cand1, cand2=cand2, cand1_name=name1, cand2_name=name2)
                captured = buf.getvalue()
                print(captured)
                with open(os.path.join(summaries_dir, f"ttest_{name1}_{name2}.txt"), "w") as f:
                    f.write(captured)
            except Exception as e:
                print(f"WARNING: ttest() failed for {name1} vs {name2} ({type(e).__name__}: {e}) -- continuing.")


if __name__ == "__main__":
    final_eval_es_setting = [
        "diss-graphs/graphs/FINAL_EVAL_ES",
        final_eval_es,
        "Configuration",
        False,
        "mlruns_csv/FINAL_EVAL_ES",
        {},
        True,
    ]
    final_eval_ns_setting = [
        "diss-graphs/graphs/FINAL_EVAL_NS",
        final_eval_ns,
        "Configuration",
        False,
        "mlruns_csv/FINAL_EVAL_NS",
        {},
        True,
    ]

    current_dataset = saga_datasets

    for setting in [final_eval_es_setting, final_eval_ns_setting]:
        mlruns_to_csv(current_dataset, subdir=setting[4].split("/")[-1], normalize=True)
        run_main()
        print(f"\nFinished creating plots for {setting[4].split('/')[-1]}")
