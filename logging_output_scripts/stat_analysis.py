# Script to perform the analysis for the 2022 paper "Approaches for Rule
# Discovery in a Learning Classifier System" by Heider et al.
#
# Copyright (C) 2022 David Pätzel <david.paetzel@posteo.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import subprocess
import sys
from functools import partial
from typing import List, Tuple

import arviz as az
import click
import cmpbayes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed
from logging_output_scripts.utils import (
    get_csv_df,
    get_dataframe,
    check_and_create_dir,
    get_all_runs,
    get_df,
    get_normalized_df,
)
import itertools
import json
import math
from utils import datasets_map

pd.options.display.max_rows = 2000
# If this doesn't work, because you can't fine Time New Roman as a font do the following:
# sudo apt install msttcorefonts -qq
# rm ~/.cache/matplotlib -rf

# TODO Store via PGF backend with nicer LaTeXy fonts etc.
# https://jwalton.info/Matplotlib-latex-PGF/
# matplotlib.use("pgf")
sns.set_theme(
    style="whitegrid",
    font="Times New Roman",
    font_scale=0.8,
    rc={"lines.linewidth": 1, "pdf.fonttype": 42, "ps.fonttype": 42},
)

# Add \the\linewidth into the LaTeX file to get this value (it's in pt).
linewidth = 213.41443
# A TeX pt is 1/72.27 inches acc. to https://tex.stackexchange.com/a/53509/191862 .
linewidth /= 72.27
# Add \the\textwidth into the LaTeX file to get this value (it's in pt).
textwidth = 449.59116
textwidth /= 72.27

elitist_complexity = "metrics.elitist_complexity"
mse = "metrics.test_neg_mean_squared_error"

metrics = {mse: "MSE", elitist_complexity: "Model Complexity"}
HIGHER_IS_BETTER = ["metrics.hypervolume", "metrics.test_hypervolume"]


def smart_print(df, latex):
    if latex:
        print(df.to_latex(index=False))
    else:
        print(df.to_markdown())


chosen_sample_num = 10000


def load_data(config):
    dfs = []
    keys = []

    for heuristic in config["heuristics"]:
        # if config["normalize_datasets"]:
        #     df = get_normalized_df(heuristic)

        for problem in config["datasets"]:
            if config["data_directory"] == "mlruns":
                df = get_df(heuristic, problem)
                df[mse] *= -1
            else:
                df = get_csv_df(heuristic, problem)

            dfs.append(df)
            keys.append((heuristic, problem))

    if config["normalize_datasets"]:
        dfs = [df.reset_index() for df in dfs]

    df = pd.concat(dfs, keys=keys, names=["algorithm", "task"], verify_integrity=True)
    df = df[config["metrics"].values()]

    if config["data_directory"] == "mlruns_csv/RBML":
        df[elitist_complexity] = 0.3

    assert not df.isna().any().any(), "Some values are missing"
    return df


def round_to_n_sig_figs(x, n):
    decimals = -int(np.floor(np.log10(np.abs(x)))) + (n - 1)
    return x if x == 0 else np.round(x, decimals)


@click.group()
def cli():
    pass


def calvo(latex=False, all_variants=False, check_mcmc=False, small_set=False, ylabel=None):

    sns.set_theme(
        style="whitegrid",
        font="Times New Roman",
        font_scale=1,
        rc={"lines.linewidth": 1, "pdf.fonttype": 42, "ps.fonttype": 42},
    )

    with open("logging_output_scripts/config.json") as f:
        config = json.load(f)

    metrics = {y: x for x, y in config["metrics"].items()}

    final_output_dir = f"{config['output_directory']}"
    # check_and_create_dir(final_output_dir, "calvo")

    df = None
    df = load_data(config)

    # Explore whether throwing away distributional information gives us any
    # insights.
    variants = {
        # Mean/median over cv runs per task (n_tasks problem instances).
        "mean of": lambda _df: _df[metric].groupby(["algorithm", "task"]).mean().unstack().T,
        "median of": lambda _df: _df[metric].groupby(["algorithm", "task"]).median().unstack().T,
        # Each cv run as a separate problem instance (n_tasks * n_cv_runs problem
        # instances).
        "all": lambda _df: _df[metric].unstack(0),
    }
    # Insight: We don't want to throw away distributional information.
    if not all_variants:
        variants = {"all": variants["all"]}

    pd.options.mode.chained_assignment = None

    for metric in metrics:
        if metric not in df or (config["data_directory"] == "mlruns_csv/RBML" and metric == elitist_complexity):
            continue
        higher_is_better = metric in HIGHER_IS_BETTER

        num_heuristics = len(config["heuristics"])
        fig, ax = plt.subplots(len(variants), figsize=(8, 0.5 * num_heuristics), dpi=72)
        plt.subplots_adjust(hspace=5)

        # fig, ax = plt.subplots(len(variants), figsize=(textwidth, 5 / 7 * 2.7), dpi=72)

        if not all_variants:
            ax = [ax]

        i = -1
        for mode, f in variants.items():
            i += 1
            d = f(df)[config["heuristics"].keys()]

            print(d)

            title = f"Considering {mode} cv runs per task"

            print(f"Sample statistics of {metrics[metric]} for “{title}” are as follows:\n")
            ranks = d.apply(np.argsort, axis=1) + 1
            print(title)
            smart_print(ranks.mean(), latex=latex)

            d.rename(columns=config["heuristics"], inplace=True)
            d = d.apply(lambda x: x.dropna().reset_index(drop=True))

            print(d.to_numpy())

            # NOTE We fix the random seed here to enable model caching.
            model = cmpbayes.Calvo(d.to_numpy(), higher_better=False, algorithm_labels=d.columns.to_list()).fit(
                num_samples=chosen_sample_num, random_seed=1
            )
            model1 = cmpbayes.Calvo(d.to_numpy(), higher_better=higher_is_better, algorithm_labels=d.columns.to_list()).fit(
                num_samples=chosen_sample_num, random_seed=1
            )

            if check_mcmc:
                smart_print(az.summary(model.infdata_), latex=latex)
                az.plot_trace(model.infdata_)
                az.plot_rank(model.infdata_)

            # Join all chains, name columns.
            sample = np.concatenate(model.infdata_.posterior.weights)
            sample = pd.DataFrame(sample, columns=model.infdata_.posterior.weights.algorithm_labels)

            xlabel = f"Probability"  # f"Probability of having the lowest {metrics[metric]}"
            sample = sample.unstack().reset_index(0).rename(columns={"level_0": ylabel, 0: xlabel})

            sns.boxplot(data=sample, y=ylabel, x=xlabel, ax=ax[i], fliersize=0.3, palette="tab10")
            ax[i].set_title(metrics[metric], style="italic")
            ax[i].set_xlabel(xlabel, weight="bold")
            ax[i].set_ylabel(ylabel, weight="bold")

        fig.tight_layout()

        if config["normalize_datasets"]:
            heuristic = list(config["heuristics"].keys())[0]
            f_index = heuristic.find("f:")
            result = heuristic[f_index + 2 :]
            result = result.replace("; -e:", "_")
            result = result.replace("/", "")

            fig.savefig(
                f"{final_output_dir}/calvo_{result}_{metric}{'' if not small_set else '-small'}.pdf",
                dpi=fig.dpi,
                bbox_inches="tight",
            )
        else:
            fig.savefig(
                f"{final_output_dir}/calvo_{metric}{'' if not small_set else '-small'}.pdf",
                dpi=fig.dpi,
                bbox_inches="tight",
            )


def cohens_pairwise_d(candidate_pairs: List[Tuple[str, str]], candidate_pair_names: List[str]) -> None:
    """
    Compute dependent-samples Cohen's d for given algorithm pairs across
    all datasets and metrics and print/save LaTeX tables.

    Parameters
    ----------
    candidate_pairs : List[Tuple[str, str]]
        List of pairs of algorithm ids (as used in the config / df index),
        e.g. [("Baseline c:ga32", "Baseline c:ga64"), ...].
    candidate_pair_names : List[str]
        List of human-readable names for each pair, in the same order as
        candidate_pairs, e.g. ["GA32 - GA64", ...].
    """
    # Basic sanity check
    if len(candidate_pairs) != len(candidate_pair_names):
        raise ValueError("candidate_pairs and candidate_pair_names must have the same length.")

    with open("logging_output_scripts/config.json") as f:
        config = json.load(f)

    metrics = {y: x for x, y in config["metrics"].items()}

    final_output_dir = f"{config['output_directory']}"
    df = load_data(config)

    pd.options.mode.chained_assignment = None

    # Ensure target directory for LaTeX tables exists
    cohens_dir = f"{final_output_dir}/tables/cohens_d"
    check_and_create_dir(final_output_dir, "tables")
    check_and_create_dir(f"{final_output_dir}/tables", "cohens_d")

    # For each metric we build a single DataFrame:
    # index = datasets, columns = candidate_pair_names
    for metric in metrics:
        comb_labels = list(candidate_pair_names)
        table_rows = []

        for task_key, _ in config["datasets"].items():
            task_name = datasets_map[task_key].upper()

            # Preload all y-values for algorithms that appear in any pair
            alg_ids = {alg for pair in candidate_pairs for alg in pair}
            y_values = {alg: df[metric].loc[alg, task_key].to_numpy() for alg in alg_ids}

            row_vals = []
            for alg1, alg2 in candidate_pairs:
                y1 = y_values[alg1]
                y2 = y_values[alg2]
                # dependent-samples Cohen's d_z: mean(diff) / sd(diff)
                diff = y1 - y2
                d_z = diff.mean() / diff.std(ddof=1)
                row_vals.append(d_z)

            table_rows.append((task_name, row_vals))

        if not table_rows:
            continue

        # Build DataFrame: rows = datasets, columns = name combinations
        data = {comb_labels[k]: [row[1][k] for row in table_rows] for k in range(len(comb_labels))}
        index = [row[0] for row in table_rows]
        d_table = pd.DataFrame(data, index=index)

        d_table.index.name = "Datasets"
        d_table = d_table.round(3)

        latex_str = d_table.to_latex(
            index=True,
            index_names=False,
            escape=False,
            float_format="%.3f".__mod__,
        )

        # Adjust column layout placeholder; length depends on number of pairs
        # Here we just replace 'lrrr' with 'l' + 'c' * n_pairs, if present.
        old_layout = "\\begin{tabular}{l" + "r" * len(comb_labels) + "}"
        new_layout = "\\begin{tabular}{l" + "c" * len(comb_labels) + "}"
        latex_str = latex_str.replace(old_layout, new_layout, 1)

        latex_str = latex_str.replace(
            "\\toprule\n & ",
            "\\toprule\nDatasets & ",
            1,
        )

        print(f"% Cohen's d (dependent) for metric: {metrics[metric]}")
        print(latex_str)
        print()

        safe_metric = metrics[metric].replace(" ", "_").replace("/", "_")
        out_path = f"{cohens_dir}/cohens_d_{safe_metric}.tex"
        with open(out_path, "w") as f_out:
            f_out.write("% Automatically generated Cohen's d table\n")
            f_out.write(f"% Metric: {metrics[metric]}\n\n")
            f_out.write(latex_str)
    return


def ttest(latex, cand1, cand2, cand1_name, cand2_name):
    with open("logging_output_scripts/config.json") as f:
        config = json.load(f)

    metrics = {y: x for x, y in config["metrics"].items()}

    final_output_dir = f"{config['output_directory']}"

    df = None
    df = load_data(config)

    pd.options.mode.chained_assignment = None

    hdis = {}
    for metric in metrics:
        hdis[metrics[metric]] = {}
        probs = {}

        print(f"# {metrics[metric]}\n")

        # fig, ax = plt.subplots(len(config["datasets"]), figsize=(textwidth if metrics[metric] == "MSE" else linewidth, 5), dpi=72)
        # fig, ax = plt.subplots(len(config["datasets"]), figsize=(textwidth, 5), dpi=72)
        num_heuristics = len(config["heuristics"])
        fig, ax = plt.subplots(
            nrows=math.ceil(len(config["datasets"]) / 2),
            ncols=2,
            figsize=(8, 0.75 * math.ceil(len(config["datasets"]) / 2) * 2),
            dpi=72,
        )
        ax = ax.ravel()
        for i, task in enumerate(config["datasets"]):
            if metric not in df or (config["data_directory"] == "mlruns_csv/RBML" and metric == elitist_complexity):
                continue
            if task in df[metric].loc[cand1]:
                y1 = df[metric].loc[cand1, task].to_numpy()
                y2 = df[metric].loc[cand2, task].to_numpy()
            # # else:
            # y2 = df[metric].loc[cand1, task].to_numpy()
            # y1 = df[metric].loc[cand2, task].to_numpy()

            model = cmpbayes.BayesCorrTTest(y1, y2, fraction_test=0.25).fit(num_samples=chosen_sample_num)

            # Compute 100(1 - alpha)% high density interval.
            alpha = 0.005
            hdi = (model.model_.ppf(alpha), model.model_.ppf(1 - alpha))
            hdis[metrics[metric]][config["datasets"][task]] = {"lower": hdi[0], "upper": hdi[1]}

            # Compute bounds of the plots based on ppf.
            xlower_ = model.model_.ppf(1e-6)
            xlower_ -= xlower_ * 0.07
            xupper_ = model.model_.ppf(1 - 1e-6)
            xupper_ += xupper_ * 0.07
            xlower = np.abs([xlower_, xupper_, *hdi]).max()
            xupper = -xlower

            # Compute pdf values of posterior.
            x = np.linspace(xlower, xupper, 1000)
            # y = model.model_.cdf(x)
            # x = np.arange(1e-3, 1 - 1e-3, 1e-3)
            y = model.model_.pdf(x)

            # xlabel = (f"{metrics[metric]}({cand2_name}) - {metrics[metric]}({cand1_name})"
            #           if metrics[metric] == "MSE"
            #           else (
            #               f"{metrics[metric].capitalize()}({cand2_name})\n- "
            #               f"{metrics[metric].capitalize()}({cand1_name})"))

            if not (i == 1 or i == 3):
                xlabel = (
                    f"MSE({cand2_name}) - MSE({cand1_name})" if metrics[metric] == "MSE" else (f"COMP({cand2_name}) - COMP({cand1_name})\n")
                )

                ylabel = "Density"
            else:
                xlabel = " "
                ylabel = "  "

            xlabel = " "
            ylabel = "  "

            data = pd.DataFrame({xlabel: x, ylabel: y})

            # Plot posterior.
            # sns.histplot(model.model_.rvs(50000),
            #              bins=100,
            #              ax=ax[i],
            #              stat="density")
            ax_val = ax[i]
            sns.lineplot(data=data, x=xlabel, y=ylabel, ax=ax_val)
            ax_val.fill_between(x, 0, y, alpha=0.33)
            ax_val.set_xlabel("")
            ax_val.set_ylabel("")
            ax_val.set_title(f"{config['datasets'][task]}", style="italic", pad=15.0)

            # Add HDI lines and values.
            ax_val.vlines(x=hdi, ymin=-0.1 * max(y), ymax=1.2 * max(y), colors="C1", linestyles="dashed")
            ax_val.text(
                x=hdi[0],
                y=1.3 * max(y),
                s=round_to_n_sig_figs(hdi[0], 2),
                ha="right",
                va="center",
                color="C1",
                fontweight="bold",
            )
            ax_val.text(
                x=hdi[1],
                y=1.3 * max(y),
                s=round_to_n_sig_figs(hdi[1], 2),
                ha="left",
                va="center",
                color="C1",
                fontweight="bold",
            )

            ax_val.set_ylim(top=1.2 * max(y))
            if metrics[metric] == "Model Complexity":
                # Compute rope for this task.
                # Remove RS runs.
                ()
                d_ = df[metric].unstack("algorithm")[[alg for alg in config["heuristics"]]].stack()
                # d_ = df[metric]  # .unstack(0).stack()
                ()
                # Rope is based on std of the other algorithms.
                stds = d_[task].groupby("algorithm").std()
                rope = stds.mean()
                rope = [-rope, rope]

                # Add rope lines and values.
                ax_val.vlines(x=rope, ymin=-0.1 * max(y), ymax=1.2 * max(y), colors="C2", linestyles="dotted")
                ax_val.fill_between(rope, 0, 1.2 * max(y), alpha=0.33, color="C2")

                # Compute probabilities.
                sample = model.model_.rvs(chosen_sample_num)

                probs[config["datasets"][task]] = {
                    f"p({cand1_name} practically higher complexity)": (sample < rope[0]).sum() / len(sample),
                    f"p(practically equivalent)": np.logical_and(rope[0] < sample, sample < rope[1]).sum() / len(sample),
                    f"p({cand2_name} practically higher complexity)": (rope[1] < sample).sum() / len(sample),
                }

            ax_val.set_ylabel(ylabel, weight="bold")
            ax_val.set_xlabel(xlabel, weight="bold")
            fig.tight_layout()

            nname = "mse" if metric == "metrics.test_neg_mean_squared_error" else "complexity"

            if metrics[metric] == "MSE":
                xlabel = f"MSE({cand2_name}) - MSE({cand1_name})"
                nname = "mse"
            elif metrics[metric] == "COMP":
                xlabel = f"COMP({cand2_name}) - COMP({cand1_name})"
                nname = "complexity"
            elif metrics[metric] == "Hypervolume":
                xlabel = f"HV({cand2_name}) - HV({cand1_name})"
                nname = "hypervolume"
            else:
                xlabel = f"{metrics[metric]}({cand2_name}) - {metrics[metric]}({cand1_name})"
                nname = metrics[metric]

            ylabel = "Density"
            fig.text(0.5, -0.01, xlabel, ha="center")
            fig.text(0.01, 0.5, ylabel, ha="center", rotation=90)

            if len(config["datasets"]) % 2 != 0:
                ax[-1].set_visible(False)

            fig.align_ylabels()
            fig.tight_layout()
            fig.savefig(f"{final_output_dir}/ttest_{cand1_name}_{cand2_name}_{nname}.pdf", dpi=fig.dpi, bbox_inches="tight")

    # https://stackoverflow.com/a/67575847/6936216
    hdis_ = hdis
    hdis_melt = pd.json_normalize(hdis_, sep=">>").melt()
    hdis = hdis_melt["variable"].str.split(">>", expand=True)
    hdis.columns = ["Metric", "Dataset", "kind"]
    hdis["bound"] = hdis_melt["value"]
    hdis = hdis.set_index(list(hdis.columns[:-1]))
    hdis = hdis.unstack("kind")
    # Format both ends first
    # Format both ends first
    lower = hdis["bound", "lower"].apply(lambda x: round_to_n_sig_figs(x, n=2))
    upper = hdis["bound", "upper"].apply(lambda x: round_to_n_sig_figs(x, n=2))

    # Merge into a single string column like "[x, y]"
    hdis[("99% HDI", "")] = "[" + lower.astype(str) + ", " + upper.astype(str) + "]"

    hdis = hdis.drop(columns=["bound"])

    hdis.columns.names = [None, None]
    hdis.columns = pd.MultiIndex.from_tuples([(col[0], "") for col in hdis.columns])
    hdis = hdis.reset_index()

    if isinstance(hdis.columns, pd.MultiIndex):
        hdis.columns = [col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in hdis.columns]

    if hdis["Metric"].nunique(dropna=True) <= 1:
        hdis = hdis.drop(columns="Metric")

    smart_print(hdis, latex=latex)

    probs = pd.DataFrame(probs).T
    probs = np.round(probs, 3)
    probs = 100 * probs
    smart_print(probs, latex=latex)


if __name__ == "__main__":
    cli()
