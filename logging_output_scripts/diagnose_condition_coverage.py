#!/usr/bin/env python3
"""
CHECK FOR ALL MLRUNS

Usage:
    python diagnose_condition_coverage.py --tracking-uri file:///path/to/mlruns
"""

import argparse
import json
import os
import re
from collections import defaultdict

import pandas as pd
from mlflow.tracking import MlflowClient

RUN_NAME_RE = re.compile(r"p:([A-Za-z_]+)\.fold-(\d+)/(\d+)")


def classify_condition(params: dict) -> str:
    rd_raw = params.get("rule_discovery") or ""
    rd_class = rd_raw.split("(")[0].strip()
    rd_method = "ns" if rd_class == "NoveltySearch" else "es"
    pruning_on = params.get("rule_discovery__subsumption") not in (None, "None")
    adaptive_on = params.get("early_stopping_patience", -1) != -1 or params.get("extra_rules_patience", -1) != -1
    return f"{rd_method}_{'pruning' if pruning_on else 'nopruning'}_" f"{'adaptive' if adaptive_on else 'noadaptive'}"


def collect_rows(tracking_uri: str) -> pd.DataFrame:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_ids = [exp.experiment_id for exp in client.search_experiments()]

    rows = []
    unparsed_names = 0
    no_params_json = 0

    for exp_id in experiment_ids:
        for run in client.search_runs(experiment_ids=[exp_id], max_results=50000):
            run_name = run.data.tags.get("mlflow.runName", "")
            m = RUN_NAME_RE.search(run_name)
            if not m:
                unparsed_names += 1
                continue
            dataset, fold_idx, fold_total = m.group(1), int(m.group(2)), int(m.group(3))

            # artifact_uri is an ABSOLUTE path from the machine that created
            # the run, so it's stale after moving mlruns/ elsewhere. Keep only
            # what's after 'suprb-experimentation/' and treat it as relative
            # to cwd, same trick moo_plots.py already uses.
            artifact_uri = run.info.artifact_uri
            if "suprb-experimentation/" in artifact_uri:
                artifact_dir = artifact_uri.split("suprb-experimentation/")[-1]
            else:
                artifact_dir = artifact_uri.replace("file://", "")
            params_path = os.path.join(artifact_dir, "params.json")
            if not os.path.exists(params_path):
                no_params_json += 1
                continue
            with open(params_path) as f:
                params = json.load(f)

            rows.append(
                {
                    "dataset": dataset,
                    "condition": classify_condition(params),
                    "seed": params.get("random_state"),
                    "fold_idx": fold_idx,
                    "fold_total": fold_total,
                    "run_id": run.info.run_id,
                    "source_dir": run.data.tags.get("source_mlruns_dir", "<no source_mlruns_dir tag>"),
                    "run_name": run_name,
                }
            )

    print(
        f"Parsed {len(rows)} fold-runs. Skipped {unparsed_names} runs with no 'p:<dataset>.fold-k/n' "
        f"in their name (root/tuning-summary runs, expected), {no_params_json} with no params.json."
    )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("No fold-runs parsed at all -- something upstream is wrong (check --tracking-uri).")
        return

    for (dataset, condition), group in sorted(df.groupby(["dataset", "condition"])):
        expected_seeds = 25
        seed_counts = group.groupby("seed").size()
        n_ok = (seed_counts == 8).sum()
        n_dup = (seed_counts > 8).sum()
        n_partial = (seed_counts < 8).sum()
        n_seeds = len(seed_counts)

        status = "OK" if n_seeds == expected_seeds and n_dup == 0 and n_partial == 0 else "MISMATCH"
        print(
            f"\n[{status}] {dataset} / {condition}: {len(group)} rows, {n_seeds} distinct seeds "
            f"(expected {expected_seeds}) -- {n_ok} clean, {n_dup} duplicated, {n_partial} incomplete"
        )

        if n_dup > 0:
            for seed, count in seed_counts[seed_counts > 8].items():
                rows = group[group["seed"] == seed]
                print(f"    DUPLICATE seed={seed} ({count} rows, expected 8):")
                for src, sub in rows.groupby("source_dir"):
                    print(f"      {len(sub):3d} rows from {src}  (folds present: {sorted(sub['fold_idx'].tolist())})")

        if n_partial > 0:
            for seed, count in seed_counts[seed_counts < 8].items():
                rows = group[group["seed"] == seed]
                present = sorted(rows["fold_idx"].tolist())
                src = rows["source_dir"].iloc[0]
                print(f"    INCOMPLETE seed={seed}: only {count}/8 folds present {present}, from {src}")

        missing_seeds = expected_seeds - n_seeds if n_seeds < expected_seeds else 0
        if missing_seeds > 0 and n_partial == 0:
            print(f"    {missing_seeds} seed(s) have NO rows at all for this (dataset, condition) pair.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    args = parser.parse_args()
    df = collect_rows(args.tracking_uri)
    report(df)
