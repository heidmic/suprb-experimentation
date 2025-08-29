"""
This file contains multiple functions to check the importance of parameters:

check_f_anova: Use fANOVA to quantify the impact of each hyperparameter and their interactions on model performance
    Usage requires the package fanova==2.0.19 (needs: sudo apt-get install swig and sudo apt-get install python3.12-dev)
        which is therefore not included in the standard requirements
check_linearity: Plot a scatter plot to visually assess linearity between two parameters
check_normality: Plot histograms to visually assess normality between two parameters
check_normality_shapiro: Use shapiro to test whether a dataset is drawn from a normal distribution (Shapiro-Wilk test)
chi2contingency: Use chi2_contingency to test for independence between two categorical variables
mutual_information: Use mutual_info_regression to estimate the dependency between each feature
"""

import re
import ast

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import defaultdict
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import shapiro, chi2_contingency
from fanova import fANOVA
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

np.random.seed(42)


def create_path(path):
    dir_path = Path(path)

    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")


# Use fANOVA to quantify the impact of each hyperparameter and their interactions on model performance
def check_f_anova(df, exclude=None):
    # Necessary for f_anova package to work with current numpy version
    np.float = float

    param_names = [col for col in df.columns if col != "value" and col != exclude]

    cs = ConfigurationSpace()
    for name in param_names:
        min_val = df[name].min()
        max_val = df[name].max()
        if min_val == max_val:
            print(f"Skipping '{name}' (constant value)")
            continue

        cs.add(UniformFloatHyperparameter(name, lower=float(min_val) - 1e-8, upper=float(max_val) + 1e-8))

    ordered_param_names = list(cs.keys())
    df = df[ordered_param_names + ["value"]]

    print("\nChecking for out-of-bound values:")
    for name in cs.keys():
        hp = cs[name]
        lower, upper = hp.lower, hp.upper
        out_of_bounds = df[(df[name] < lower) | (df[name] > upper)]
        if not out_of_bounds.empty:
            print(f"'{name}' has {len(out_of_bounds)} out-of-bound values (Range: {lower}, {upper}):")
            print(out_of_bounds[[name]])

    X = df[ordered_param_names].to_numpy()
    y = df["value"].to_numpy()

    fanova = fANOVA(X, y, config_space=cs)
    importances_list = []

    for i, name in enumerate(ordered_param_names):
        imp = fanova.quantify_importance((i,))
        importance_value = imp[(i,)]["individual importance"]
        importances_list.append((name, importance_value))

    importances_df = pd.DataFrame(importances_list, columns=["Feature", "Importance"])

    importances_df = importances_df.sort_values(by="Importance", ascending=False)

    print(importances_df.to_string(index=False, float_format="%.4f"))

    plt.barh(importances_df["Feature"], importances_df["Importance"])
    plt.xlabel("Importance")
    plt.title("fANOVA Feature Importances")
    plt.tight_layout()

    plt.show()


# Plot a scatter plot to visually assess linearity between two parameters
def check_linearity(df, col1, col2):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[col1], y=df[col2])
    plt.title(f"Scatter Plot of {col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)

    directory = "logging_output_scripts/outputs/importance"
    create_path(directory)
    plt.savefig(f"{directory}/linearity_{col1}_{col2}")


# Plot histograms to visually assess normality between two parameters
def check_normality(df, col1, col2):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col1], kde=True, color="blue", label=col1)
    sns.histplot(df[col2], kde=True, color="red", label=col2)
    plt.legend()
    plt.title(f"Histogram and KDE of {col1} and {col1}")

    directory = "logging_output_scripts/outputs/importance"
    create_path(directory)
    plt.savefig(f"{directory}/normality_{col1}_{col2}")


# Use shapiro to test whether a dataset is drawn from a normal distribution (Shapiro-Wilk test)
def check_normality_shapiro(data):
    stat, p_value = shapiro(data)

    print(f"Test Statistic: {stat}")
    print(f"P-value: {p_value}")

    if p_value > 0.05:
        print("The data is likely normally distributed (fail to reject H0).")
    else:
        print("The data is likely not normally distributed (reject H0).")


# Use chi2_contingency to test for independence between two categorical variables
def chi2contingency(df, col1, col2):
    contingency_table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Expected Frequencies:\n{expected}")

    if (expected < 5).any():
        print("Warning: Some expected frequencies are less than 5, which may violate the Chi-Square assumption.")
    else:
        print("All expected frequencies are adequate.")


# Use mutual_info_regression to estimate the dependency between each feature
def mutual_information(X, y, x_cols):
    mi_val = mutual_info_regression(X, y)
    data = list(zip(x_cols, mi_val))
    result = pd.DataFrame(data, columns=["parameter", "fitness"])

    result = result.round(3)
    result = result.applymap(lambda x: str(x).replace(".", ","))

    directory = "logging_output_scripts/outputs/importance"
    create_path(directory)

    result.to_csv(f"{directory}/mutual_information.csv", sep=";", index=False)


def get_df(f):
    data = []

    def extract_trial_lines(file_path):
        pattern = re.compile(r"\bTrial\s+\d+\s+finished with value:")
        matched_lines = []

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if pattern.search(line):
                    matched_lines.append(line.strip())  # strip() removes newlines

        return matched_lines

    log_text = extract_trial_lines(f)

    for line in log_text:
        trial_match = re.search(r"Trial (\d+)", line)
        if trial_match:
            trial_number = int(trial_match.group(1))

        value_match = re.search(r"value: (-?\d+\.\d+)", line)
        if value_match:
            value = float(value_match.group(1))

        params_match = re.search(r"parameters: ({.*?})", line)
        if params_match:
            parameters = ast.literal_eval(params_match.group(1))
            parameters["trial"] = trial_number
            parameters["value"] = value

        data.append(parameters)

    df = pd.DataFrame(data)
    df = df[(df["value"] > 0)]
    # df = df[(df["value"] > 0) & (df["value"] < 1)]

    df = df[df["rule_discovery"] == "ES1xLambda"]
    df = df[df["solution_composition"] == "GeneticAlgorithm"]

    df = df.dropna(subset=["value"])
    df = df.drop(columns=["trial"])

    return df


def get_merged_df(files):
    df = pd.DataFrame()
    for f in files:
        df = pd.concat([df, get_df(f)])

    categorical_cols = [
        "rule_discovery__acceptance",
        "rule_discovery__constraint",
        "rule_discovery__init",
        "rule_discovery__init__fitness",
        "rule_discovery__mutation",
        "rule_discovery__mutation__mutation",
        "rule_discovery__operator",
        "rule_discovery__origin_generation",
        "rule_discovery__selection",
        "solution_composition__crossover",
        "solution_composition__init",
        "solution_composition__init__fitness",
        "solution_composition__init__mixing__experience_calculation",
        "solution_composition__init__mixing__filter_subpopulation",
        "solution_composition__selection",
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    df_cleaned = df.replace([np.inf, -np.inf], np.nan)
    df = df_cleaned.dropna(axis=1)

    X = df.drop(columns=["value", "rule_discovery", "solution_composition"])
    y = df["value"].to_numpy()

    x_cols = X.columns

    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape((-1, 1))).reshape((-1,))

    df.columns = df.columns.str.replace("rule_discovery", "rd")
    df.columns = df.columns.str.replace("solution_composition", "sc")
    df = df.drop(columns=["rd", "sc"], errors="ignore")

    X_scaled_df = pd.DataFrame(X, columns=x_cols)

    return df, X, y, X_scaled_df, x_cols


def calculate_numeric_ranges(top_k_runs):
    numeric_cols = top_k_runs.select_dtypes(include="number")
    numeric_summary = pd.DataFrame(
        {
            "min": numeric_cols.min(),
            "max": numeric_cols.max(),
            "median": numeric_cols.median(),
            "5% quantil": numeric_cols.quantile(0.05),
            "95% quantil": numeric_cols.quantile(0.95),
        }
    )

    print(numeric_summary)


def calculate_non_numeric_ranges(top_k_runs):
    bool_cols = top_k_runs.select_dtypes(include="bool")

    extract_deep_prefix = lambda col: (
        col if "__" not in col else f"{col.rsplit('__', 1)[0]}__{col.rsplit('__', 1)[1].split('_')[0]}",
        col if "__" not in col else col.rsplit("__", 1)[1],
    )

    prefix_to_postfix_stats = defaultdict(list)

    for col in bool_cols.columns:
        deep_prefix, postfix = extract_deep_prefix(col)
        true_rows = top_k_runs[top_k_runs[col]]
        count_true = len(true_rows)

        if count_true > 0:
            prefix_to_postfix_stats[deep_prefix].append(
                {
                    "postfix": postfix,
                    "true_count": count_true,
                    "min": true_rows["value"].min(),
                    "max": true_rows["value"].max(),
                    "median": true_rows["value"].median(),
                    "5% quantil": true_rows["value"].quantile(0.05),
                    "95% quantil": true_rows["value"].quantile(0.95),
                }
            )

    for prefix, rows in prefix_to_postfix_stats.items():
        print(f"\n\033[31m{prefix}: \033[0m")
        df_group = pd.DataFrame(rows).dropna()
        total_true = df_group["true_count"].sum()
        df_group["percentage_chosen"] = (df_group["true_count"] / total_true * 100).round(2).astype(str) + "%"
        df_group = df_group.drop(columns=["true_count"])
        cols = ["percentage_chosen"] + [col for col in df_group.columns if col != "percentage_chosen"]
        df_group = df_group[cols]

        print(df_group.to_string(index=False))


def calculate_ranges_from_tuning_results(df, k_largest=None, k_percentage=None):
    if k_largest:
        top_k_runs = df.nlargest(k_largest, "value")
    elif k_percentage:
        top_k_runs = df.nlargest(int(len(df) * 0.10), "value")
    else:
        print("You need to provide either k_largest or k_percentage!")
        return

    calculate_numeric_ranges(top_k_runs)
    calculate_non_numeric_ranges(top_k_runs)


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)

    files = [
        # PseudoBIC
        "anova_output/output-8183338.txt",
        # "anova_output/output-8183339.txt",
        # "anova_output/output-8183340.txt",
        # "anova_output/output-8183341.txt",
        # # WU
        # "anova_output/output-8183343.txt",
        # "anova_output/output-8183344.txt",
        # "anova_output/output-8183346.txt",
        # "anova_output/output-8183347.txt",
    ]

    df, X, y, X_scaled_df, x_cols = get_merged_df(files)

    calculate_ranges_from_tuning_results(df, k_largest=20)

    # check_f_anova(df)
    # chi2contingency(df, "sc__crossover_rate", "sc__mutation_rate")
    # check_linearity(df, "sc__crossover_rate", "sc__mutation_rate")
    # check_normality(df, "sc__crossover_rate", "sc__mutation_rate")
    # check_normality_shapiro(X_scaled_df)
    # mutual_information(X, y, x_cols)
