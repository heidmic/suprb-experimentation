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

import arviz as az
import click
import cmpbayes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython import embed
from logging_output_scripts.utils import get_dataframe, check_and_create_dir, get_all_runs, get_df
import json


pd.options.display.max_rows = 2000

# TODO Store via PGF backend with nicer LaTeXy fonts etc.
# https://jwalton.info/Matplotlib-latex-PGF/
# matplotlib.use("pgf")
sns.set_theme(style="whitegrid",
              font="Times New Roman",
              font_scale=0.8,
              rc={
                  "lines.linewidth": 1,
                  "pdf.fonttype": 42,
                  "ps.fonttype": 42
              })

# Add \the\linewidth into the LaTeX file to get this value (it's in pt).
linewidth = 213.41443
# A TeX pt is 1/72.27 inches acc. to https://tex.stackexchange.com/a/53509/191862 .
linewidth /= 72.27
# Add \the\textwidth into the LaTeX file to get this value (it's in pt).
textwidth = 449.59116
textwidth /= 72.27

elitist_complexity = "metrics.elitist_complexity"
mse = "metrics.test_neg_mean_squared_error"

metrics = {
    mse: "MSE",
    elitist_complexity: "model complexity"
}


def smart_print(df, latex):
    if latex:
        print(df.to_latex())
    else:
        print(df.to_markdown())


def load_data(config):
    dfs = []
    keys = []
    # all_runs_list = get_all_runs()

    for heuristic in config['heuristics']:
        for problem in config['datasets']:
            # df = get_dataframe(all_runs_list, heuristic, problem)
            df = get_df(heuristic, problem)
            if df is not None:
                df[mse] *= -1
                dfs.append(df)
                keys.append((heuristic, problem))

    df = pd.concat(dfs, keys=keys, names=["algorithm", "task"], verify_integrity=True)
    df = df[metrics.keys()]

    assert not df.isna().any().any(), "Some values are missing"
    return df


def round_to_n_sig_figs(x, n):
    decimals = -int(np.floor(np.log10(np.abs(x)))) + (n - 1)
    return x if x == 0 else np.round(x, decimals)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--latex/--no-latex",
              help="Generate LaTeX output (tables etc.)",
              default=True)
@click.option("--all-variants/--no-all-variants",
              help="Plot different variants for interpreting run/cv data",
              default=False)
@click.option(
    "--check-mcmc/--no-check-mcmc",
    help="Whether to show plots/tables for rudimentary MCMC sanity checking",
    default=False)
@click.option("--small-set/--no-small-set",
              help="Whether to only analyse ES, NS, NSLC",
              default=False)
def calvo(latex = False, all_variants = False, check_mcmc = False, small_set = False):
    with open('logging_output_scripts/config.json') as f:
        config = json.load(f)

    final_output_dir = f"{config['output_directory']}"
    check_and_create_dir(final_output_dir, "calvo")

    df = load_data(config)

    # Explore whether throwing away distributional information gives us any
    # insights.
    variants = ({
        # Mean/median over cv runs per task (n_tasks problem instances).
        "mean of":
        lambda _df: _df[metric].groupby(["algorithm", "task"]).mean().unstack().T,
        "median of":
        lambda _df: _df[metric].groupby(["algorithm", "task"]).median().
        unstack().T,
        # Each cv run as a separate problem instance (n_tasks * n_cv_runs problem
        # instances).
        "all":
        lambda _df: _df[metric].unstack(0),
    })
    # Insight: We don't want to throw away distributional information.
    if not all_variants:
        variants = {"all": variants["all"]}

    pd.options.mode.chained_assignment = None

    for metric in metrics:
        fig, ax = plt.subplots(len(variants), figsize=(
            textwidth, 2.7 if metrics[metric] == "MSE" else 5 / 7 * 2.7), dpi=72)

        if not all_variants:
            ax = [ax]

        i = -1
        for mode, f in variants.items():
            i += 1
            d = f(df)[config["heuristics"]]

            title = f"Considering {mode} cv runs per task"

            print(f"Sample statistics of {metrics[metric]} for “{title}” are as follows:\n")
            ranks = d.apply(np.argsort, axis=1) + 1
            print(title)
            smart_print(ranks.mean(), latex=latex)

            d.rename(columns=config["heuristics"], inplace=True)

            # NOTE We fix the random seed here to enable model caching.
            model = cmpbayes.Calvo(
                d.to_numpy(),
                higher_better=False, algorithm_labels=d.columns.to_list()).fit(num_samples=10000, random_seed=1)

            if check_mcmc:
                smart_print(az.summary(model.infdata_), latex=latex)
                az.plot_trace(model.infdata_)
                az.plot_rank(model.infdata_)

            # Join all chains, name columns.
            sample = np.concatenate(model.infdata_.posterior.weights)
            sample = pd.DataFrame(sample, columns=model.infdata_.posterior.weights.algorithm_labels)

            xlabel = f"Probability"  # f"Probability of having the lowest {metrics[metric]}"
            ylabel = "RD method"
            sample = sample.unstack().reset_index(0).rename(columns={"level_0": ylabel, 0: xlabel})

            sns.boxplot(data=sample, y=ylabel, x=xlabel, ax=ax[i], fliersize=0.3)
            ax[i].set_title(metrics[metric], style="italic")
            ax[i].set_xlabel(xlabel, weight="bold")
            ax[i].set_ylabel(ylabel, weight="bold")

        fig.tight_layout()
        fig.savefig(f"{final_output_dir}/calvo/{metric}{'' if not small_set else '-small'}.pdf",
                    dpi=fig.dpi, bbox_inches="tight")


@ cli.command()
@ click.option("--latex/--no-latex",
               help="Generate LaTeX output (tables etc.)",
               default=True)
@ click.argument("cand1",
                 default="")
@ click.argument("cand2",
                 default="")
@ click.argument("cand1_name",
                 default="")
@ click.argument("cand2_name",
                 default="")
def ttest(latex, cand1, cand2, cand1_name, cand2_name):
    check_and_create_dir(final_output_dir, "ttest")
    df = load_data()
    pd.options.mode.chained_assignment = None

    hdis = {}
    for metric in metrics:
        hdis[metrics[metric]] = {}
        probs = {}

        print(f"# {metrics[metric]}\n")

        fig, ax = plt.subplots(4, figsize=(textwidth if metrics[metric] == "MSE" else linewidth, 5), dpi=72)
        for i, task in enumerate(config["datasets"]):
            if task in df[metric].loc[cand1]:
                y1 = df[metric].loc[cand1, task]
                y2 = df[metric].loc[cand2, task]
                model = cmpbayes.BayesCorrTTest(y1, y2, fraction_test=0.25).fit()

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

                xlabel = (f"{metrics[metric]}({cand2_name}) - {metrics[metric]}({cand1_name})"
                          if metrics[metric] == "MSE"
                          else (
                              f"{metrics[metric].capitalize()}({cand2_name})\n- "
                              f"{metrics[metric].capitalize()}({cand1_name})"))

                ylabel = "Density"
                data = pd.DataFrame({xlabel: x, ylabel: y})

                # Plot posterior.
                # sns.histplot(model.model_.rvs(50000),
                #              bins=100,
                #              ax=ax[i],
                #              stat="density")
                sns.lineplot(data=data, x=xlabel, y=ylabel, ax=ax[i])
                ax[i].fill_between(x, 0, y, alpha=0.33)
                ax[i].set_xlabel("")
                ax[i].set_ylabel("")
                ax[i].set_title(f"{config['datasets'][task]}", style="italic")

                # Add HDI lines and values.
                ax[i].vlines(x=hdi, ymin=-0.1 * max(y), ymax=1.2 * max(y), colors="C1", linestyles="dashed")
                ax[i].text(x=hdi[0], y=1.3 * max(y), s=round_to_n_sig_figs(hdi[0], 2),
                           ha="right", va="center", color="C1", fontweight="bold")
                ax[i].text(x=hdi[1], y=1.3 * max(y), s=round_to_n_sig_figs(hdi[1], 2),
                           ha="left", va="center", color="C1", fontweight="bold")

                ax[i].set_ylim(top=1.2 * max(y))
                if metrics[metric] == elitist_complexity:
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
                    ax[i].vlines(x=rope, ymin=-0.1 * max(y), ymax=1.2 * max(y), colors="C2", linestyles="dotted")
                    ax[i].fill_between(rope, 0, 1.2 * max(y), alpha=0.33, color="C2")

                    # Compute probabilities.
                    sample = model.model_.rvs(100000)

                    probs[config['datasets'][task]] = {
                        "p(ES practically higher complexity)": (sample < rope[0]).sum() / len(sample),
                        "p(practically equivalent)": np.logical_and(rope[0] < sample, sample < rope[1]).sum() / len(sample),
                        "p(NSLC practically higher complexity)": (rope[1] < sample).sum() / len(sample)}

                ax[i].set_ylabel(ylabel, weight="bold")
                ax[i].set_xlabel(xlabel, weight="bold")
                fig.tight_layout()
                fig.savefig(f"{final_output_dir}/ttest/{metric}.pdf", dpi=fig.dpi, bbox_inches="tight")

    # https://stackoverflow.com/a/67575847/6936216
    hdis_ = hdis
    hdis_melt = pd.json_normalize(hdis_, sep=">>").melt()
    hdis = hdis_melt["variable"].str.split(">>", expand=True)
    hdis.columns = ["n", "metric", "task", "kind"]
    del hdis["n"]
    hdis["bound"] = hdis_melt["value"]
    hdis = hdis.set_index(list(hdis.columns[:-1]))
    hdis = hdis.unstack("kind")
    hdis["bound", "lower"] = hdis["bound", "lower"].apply(lambda x: f"[{round_to_n_sig_figs(x, n=2)},")
    hdis["bound", "upper"] = hdis["bound", "upper"].apply(lambda x: f"{round_to_n_sig_figs(x, n=2)}]")
    hdis = hdis.rename(columns={"bound": "99\% HDI"})

    smart_print(hdis, latex=latex)

    probs = pd.DataFrame(probs).T
    probs = np.round(probs, 3)
    probs = 100 * probs
    smart_print(probs, latex=latex)


if __name__ == "__main__":
    cli()
