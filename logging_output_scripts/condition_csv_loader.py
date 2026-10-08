import json
import os
import re

import pandas as pd

KNOWN_MISSING = set()

_FOLD_SUFFIX = re.compile(r"\.fold-\d+/\d+$")


def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist -- did you run mlruns_to_csv for this dataset/subdir yet?")
    return pd.read_csv(path)


def get_condition_df(problem: str, condition: str, subdir: str) -> pd.DataFrame:
    path = os.path.join("mlruns_csv", subdir, f"{problem}_all.csv")
    df = _read(path)
    if "tags.condition" not in df.columns:
        raise KeyError(
            f"'tags.condition' column missing from {path}. Make sure annotate_conditions.py ran "
            f"BEFORE mlruns_to_csv, and that mlruns_to_csv's exported columns include 'tags.condition'."
        )
    out = df[df["tags.condition"] == condition].reset_index(drop=True)
    if out.empty:
        if (problem, condition) in KNOWN_MISSING:
            print(f"WARNING: no data for condition={condition!r} on {problem!r} -- listed in KNOWN_MISSING, " f"continuing without it.")
            return df.iloc[0:0].copy()
        available = sorted(df["tags.condition"].dropna().unique().tolist())
        raise ValueError(f"No rows with tags.condition == {condition!r} in {path}. Available: {available}")
    return out


def get_condition_root_df(problem: str, condition: str, subdir: str) -> pd.DataFrame:

    roots = _read(os.path.join("mlruns_csv", subdir, f"{problem}_roots.csv"))

    if "tags.condition" in roots.columns:
        tagged = roots[roots["tags.condition"] == condition]
        if not tagged.empty:
            return tagged.reset_index(drop=True)

    folds = get_condition_df(problem, condition, subdir)
    if folds.empty:
        return roots.iloc[0:0].copy()

    prefixes = set(folds["tags.mlflow.runName"].astype(str).str.replace(_FOLD_SUFFIX, "", regex=True))
    matched = roots[roots["tags.mlflow.runName"].astype(str).isin(prefixes)]
    if matched.empty:
        print(f"WARNING: no root run matched condition={condition!r} on {problem!r} -- " f"tuning table for it will be empty.")
    return matched.reset_index(drop=True)


RAW_TO_BARE = {
    "metrics.elitist_complexity": "elitist_complexity",
    "metrics.test_neg_mean_squared_error": "elitist_error",
}


def get_csv_df_for_condition(heuristic: str, problem: str) -> pd.DataFrame:

    with open("logging_output_scripts/config.json") as f:
        config = json.load(f)
    subdir = config["data_directory"].split("/")[-1]
    df = get_condition_df(problem, heuristic, subdir)
    df = df.rename(columns=RAW_TO_BARE)

    rename_map = config["metrics"]
    return df.rename(columns=rename_map)
