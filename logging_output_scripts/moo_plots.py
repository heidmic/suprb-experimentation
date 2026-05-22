from logging_output_scripts.utils import get_csv_df, get_df, get_csv_root_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import re
import os
from utils import datasets_map
from sklearn.linear_model import LinearRegression
from suprb.logging.metrics import spread as metric_spread, hypervolume as metric_hypervolume
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from typing import Dict, List, Tuple, Any, Union, Optional
import ast

mse = "metrics.test_neg_mean_squared_error"
complexity = "metrics.elitist_complexity"
hypervolume = "metrics.hypervolume"
spread = "metrics.spread"

ga_baselines = {
    "Baseline c:ga32": ("x", "red"),
    "Baseline c:ga64": ("+", "green"),
}

max_rule_pool_size = 128

# Mapping raw parameter keys to nicer LaTeX display names
PARAM_NAME_MAP = {
    "rule_discovery__mutation__sigma": "\\acs{RD} $\\sigma_{mut}$",
    "rule_discovery__init__fitness__alpha": "\\acs{RD} $\\alpha_{init}$",
    "solution_composition__crossover": "\\acs{SC} Crossover",
    "solution_composition__crossover__n": "\\acs{SC} $n_{cross}$",
    "solution_composition__mutation__mutation_rate": "\\acs{SC} $\\sigma_{mut}$",
    "solution_composition__sampler__a": "$a$",
    "solution_composition__sampler__b": "$b$",
    "solution_composition__algorithm_1__crossover": "\\acs{GA} Crossover",
    "solution_composition__algorithm_1__crossover__n": "\\acs{GA} $n_{cross}$",
    "solution_composition__algorithm_1__mutation_rate": "\\acs{GA} $\\sigma_{mut}$",
    "solution_composition__algorithm_1__selection__k": "\\acs{GA} $k_{sel}$",
    "solution_composition__algorithm_2__crossover": "\\acs{MOO} Crossover",
    "solution_composition__algorithm_2__crossover__n": "\\acs{MOO} $n_{cross}$",
    "solution_composition__algorithm_2__mutation_rate": "\\acs{MOO} $\\sigma_{mut}$",
    "solution_composition__algorithm_2__selection__k": "\\acs{MOO} $k_{sel}$",
}


def _esc_tex(s: str) -> str:
    return str(s).replace("_", r"\_")


def c_pareto_sacrifice(moo_front: np.ndarray, ga_front: np.ndarray) -> float:
    """
    Compute the c distance from the one GA solution to the solution along the MOO Pareto front with the minimum
    c value whose f2 value is smaller than the GA solution's e value. If no MOO solution has e smaller than the GA solution's e,
    choose the MOO solution with the smallest e.
    """
    if len(moo_front) == 0 or len(ga_front) == 0:
        return np.nan
    ga_sol = ga_front[0]
    ga_f2 = ga_sol[1]
    # Filter MOO front to only those with f2 less than ga_f2
    filtered_moo = moo_front[moo_front[:, 1] <= ga_f2]
    if len(filtered_moo) == 0:
        return np.nan
    else:
        closest_moo = filtered_moo[np.argmin(filtered_moo[:, 0])]
        return (closest_moo[0] - ga_sol[0]) * max_rule_pool_size


def e_pareto_sacrifice(moo_front: np.ndarray, ga_front: np.ndarray) -> float:
    """
    Compute the e distance from the one GA solution to the solution along the MOO Pareto front with the minimum
    e value whose f1 value is smaller than the GA solution's c value. If no MOO solution has c smaller than the GA solution's c,
    choose the MOO solution with the smallest c.
    """
    if len(moo_front) == 0 or len(ga_front) == 0:
        return np.nan
    ga_sol = ga_front[0]
    ga_f1 = ga_sol[0]
    # Filter MOO front to only those with f1 less than ga_f1
    filtered_moo = moo_front[moo_front[:, 0] <= ga_f1]
    if len(filtered_moo) == 0:
        return np.nan
    else:
        closest_moo = filtered_moo[np.argmin(filtered_moo[:, 1])]
        return closest_moo[1] - ga_sol[1]


def generate_pareto_sacrifice_undefined_table(
    per_problem_nan: Dict[str, Dict[str, Optional[float]]],
    cfg: Dict[str, Any],
    moo_algos: List[str],
    final_output_dir: str,
) -> None:
    """
    One LaTeX table: rows=datasets, cols=moo_algos, values=\% of NaNs in $c$ Pareto sacrifice
    """
    # Ensure tables directory (top-level, no 'supplementary' folder)
    tables_dir = os.path.join(final_output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    header = ["Dataset"] + moo_algos
    col_format = "l" + "c" * len(moo_algos)

    lines = [
        r"\begin{tabular}{" + col_format + r"}",
        r"\hline",
        " & ".join([_esc_tex(h) for h in header]) + r" \\",
        r"\hline",
    ]

    # Keep dataset order from cfg
    for problem in cfg["datasets"]:
        ds_name = _esc_tex(datasets_map[problem].upper())
        stats = per_problem_nan.get(problem, {})
        row_vals: List[str] = []
        for algo in moo_algos:
            v = stats.get(algo, None)
            cell = "N/A" if v is None or np.isnan(v) else f"{v:.1f}~\\%"
            row_vals.append(cell)
        lines.append(ds_name + " & " + " & ".join(row_vals) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]

    out_path = os.path.join(tables_dir, "pareto_sacrifice_undefined.tex")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def generate_tuning_tables(
    tuning_params: Dict[str, Dict[str, Dict]], config: Dict[str, Any], final_output_dir: str
) -> None:
    """
    Generate one LaTeX table per heuristic (renamed) showing tuned parameter values across datasets.
    Rows: datasets
    Columns: union of all parameter names observed for that heuristic across datasets.
    Missing values are filled with N/A.
    """
    # Write tuning tables into top-level tables dir (no 'supplementary')
    tuning_dir = os.path.join(final_output_dir, "tables/tuning_results")
    os.makedirs(tuning_dir, exist_ok=True)
    heuristic_display_map = config["heuristics"]

    # Collect all heuristic keys encountered
    heuristic_keys = set()
    for problem_dict in tuning_params.values():
        heuristic_keys.update(problem_dict.keys())

    for heuristic_key in sorted(heuristic_keys):
        display_name = heuristic_display_map.get(heuristic_key, heuristic_key)

        # Build an ordered union of parameter keys (order of first appearance)
        seen = {}
        for problem in config["datasets"]:
            params_dict = tuning_params.get(problem, {}).get(heuristic_key, {})
            for k in params_dict.keys():
                if k not in seen:
                    seen[k] = None
        if not seen:
            continue
        param_keys = list(seen.keys())

        param_keys_display = [PARAM_NAME_MAP.get(k, k) for k in param_keys]
        param_keys, param_keys_display = zip(*sorted(zip(param_keys, param_keys_display), key=lambda x: x[1]))
        param_keys = list(param_keys)
        param_keys_display = list(param_keys_display)
        header_cols = ["Dataset"] + param_keys_display
        col_format = "l" + "c" * len(param_keys)

        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\caption{Tuned parameters for " + _esc_tex(display_name) + r"}",
            r"\adjustbox{max width=\textwidth}{" r"\begin{tabular}{" + col_format + r"}",
            r"\hline",
            " & ".join(header_cols) + r" \\",
            r"\hline",
        ]

        for problem in config["datasets"]:
            dataset_name = datasets_map[problem].upper()
            params_dict = tuning_params.get(problem, {}).get(heuristic_key, {})
            row_vals = []
            for k in param_keys:
                if k in params_dict:
                    v = params_dict[k]
                    if isinstance(v, float):
                        v = f"{v:.4f}"
                else:
                    v = "N/A"
                row_vals.append(_esc_tex(v))
            lines.append(_esc_tex(dataset_name) + " & " + " & ".join(row_vals) + r" \\")
        lines += [r"\hline", r"\end{tabular}}", r"\end{table}"]

        content = "\n".join(lines)
        file_safe = display_name.lower().replace(" ", "_")
        out_path = os.path.join(tuning_dir, f"{file_safe}_tuned_params.tex")
        with open(out_path, "w") as f:
            f.write(content)


def configure_style() -> None:
    """
    Creating Hexbin, Kernel density estimate, and Pareto Front cardinality histogram plots.
    """
    sns.set_style("whitegrid")
    sns.set_theme(
        style="whitegrid",
        font="Times New Roman",
        font_scale=1.7,
        rc={"lines.linewidth": 1, "pdf.fonttype": 42, "ps.fonttype": 42},
    )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["figure.dpi"] = 200


def load_config(path: str = "logging_output_scripts/config.json") -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def confidence_ellipse(mean, cov, ax, n_std=1.96, color="red", **kwargs):
    """
    https://de.matplotlib.net/stable/gallery/statistics/confidence_ellipse.html#the-plotting-function-itself
    http://www.econ.uiuc.edu/~roger/courses/471/lectures/L5.pdf

    Create a plot of the covariance confidence ellipse with mean and cov*.

    Parameters
    ----------
    mean: array-like, shape (2,)
        mean of the data point
    cov: array-like, shape (2,2)
        covariance matrix
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=color, edgecolor=color, **kwargs
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    ellipse.set_edgecolor(color)
    ellipse.set_facecolor(
        (ellipse.get_facecolor()[0], ellipse.get_facecolor()[1], ellipse.get_facecolor()[2], 0.2)
    )  # 20% alpha for fill
    ax.add_patch(ellipse)
    return ellipse


def load_fold_dataframe(heuristic: str, problem: str, cfg: Dict[str, Any]) -> Union[pd.DataFrame, None]:
    if cfg["data_directory"] == "mlruns":
        return get_df(heuristic, problem)
    return get_csv_df(heuristic, problem), get_csv_root_df(heuristic, problem)


def load_roof_dataframe(heuristic: str, problem: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    if cfg["data_directory"] == "mlruns":
        return get_df(heuristic, problem)
    return get_csv_df(heuristic, problem)


def process_artifact_paths(
    current_res: pd.DataFrame,
    renamed_heuristic: str,
    train_pareto_fronts: Dict[str, List[List]],
    train_pareto_solutions: Dict[str, List[List[float]]],
    test_pareto_fronts: Dict[str, List[List]],
    test_pareto_solutions: Dict[str, List[List[float]]],
) -> None:
    for path in current_res["artifact_uri"]:
        path = path.split("suprb-experimentation/")[-1]
        with open(os.path.join(path, "pareto_fronts.json")) as f:
            train_pf = json.load(f)
            train_pf = train_pf[str(max(int(key) for key in train_pf.keys()))]
            train_pareto_fronts[renamed_heuristic].append(train_pf)
            train_pareto_solutions[renamed_heuristic].extend(train_pf)
        with open(os.path.join(path, "test_pareto_front.json")) as f:
            test_pf = json.load(f)
            test_pf = test_pf["test_pf_fitness"]
            test_pareto_fronts[renamed_heuristic].append(test_pf)
            test_pareto_solutions[renamed_heuristic].extend(test_pf)


def prepare_soo_stats(
    pareto_solutions: Dict[str, np.ndarray], cfg: Dict[str, Any]
) -> Tuple[Dict[str, Tuple[np.ndarray, Tuple[str, str]]], Dict[str, Tuple[np.ndarray, Tuple[str, str]]]]:
    soo_heuristics = [
        (cfg["heuristics"][algo], ga_baselines[algo])
        for algo in cfg["heuristics"].keys()
        if algo in ga_baselines.keys()
    ]
    soo_averages: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    soo_standard_devs: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    for algo, style in soo_heuristics:
        if len(pareto_solutions[algo]) > 0:
            soo_averages[algo] = (np.mean(pareto_solutions[algo], axis=0), style)
            soo_standard_devs[algo] = (np.cov(pareto_solutions[algo].T), style)
    return soo_averages, soo_standard_devs


def determine_layout(n_algs: int) -> Tuple[int, int]:
    # Plotting Setup
    if n_algs <= 3:
        return n_algs, 1
    if n_algs == 4:
        return 2, 2
    n_cols = 3
    n_rows = (n_algs - 1) // 3 + 1
    return n_cols, n_rows


def plot_hexbin(
    moo_heuristics: List[str],
    pareto_solutions: Dict[str, np.ndarray],
    soo_averages: Dict[str, Tuple[np.ndarray, Tuple[str, str]]],
    soo_standard_devs: Dict[str, Tuple[np.ndarray, Tuple[str, str]]],
    n_cols: int,
    n_rows: int,
    title: str,
    final_output_dir: str,
    dataset_key: str,
    plot_type: str = "test",
) -> None:
    """
    Create hexbin plot for Pareto solutions.

    Parameters:
    -----------
    pareto_solutions: Dict[str, np.ndarray]
        Dictionary mapping algorithm names to their Pareto solution arrays
    plot_type: str
        String identifier to add to plot title and filename (e.g., "test", "train")
    """
    fig_hex, axes_hex = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharex=True, sharey=True, constrained_layout=True, squeeze=False
    )

    # Add plot_type to title if it's not "test" (for backward compatibility)
    display_title = f"{title}" if plot_type == "test" else f"{title} ({plot_type.capitalize()})"
    fig_hex.suptitle(display_title)

    hb_list = []  # collect hexbin artists so we can unify their scale afterwards
    gridsize = 30

    for i, algo in enumerate(moo_heuristics):
        # Guard against empty arrays
        data = pareto_solutions.get(algo, np.array([]))
        if data is None or data.size == 0:
            # Leave axis empty but labeled
            axes_hex[i % n_rows, i // n_rows].set_title(f"{algo}")
            axes_hex[i % n_rows, i // n_rows].set_xlabel("$f_1$")
            continue

        algo_df = pd.DataFrame(
            {"Normed Complexity": pareto_solutions[algo][:, 0], "Pseudo Accuracy": pareto_solutions[algo][:, 1]}
        )
        hb = axes_hex[i % n_rows, i // n_rows].hexbin(
            algo_df["Normed Complexity"],
            algo_df["Pseudo Accuracy"],
            gridsize=gridsize,
            cmap="Blues" if plot_type == "test" else "Oranges",
            extent=(0, 1, 0, 1) if len(moo_heuristics) != 1 else None,
            mincnt=1,
        )
        hb_list.append(hb)
        axes_hex[i % n_rows, i // n_rows].set_title(f"{algo}")
        # Plot SOO comparison points
        for soo_algo, value in soo_averages.items():
            avg, style = value
            cov = soo_standard_devs[soo_algo][0]
            axes_hex[i % n_rows, i // n_rows].plot(
                avg[0],
                avg[1],
                marker=style[0],
                markersize=6,
                markeredgewidth=1.2,
                color=style[1],
                linestyle="None",
                label=f"{soo_algo} Mean",
            )
            confidence_ellipse(
                avg,
                cov,
                axes_hex[i % n_rows, i // n_rows],
                n_std=1.96,
                color=style[1],
                label=f"{soo_algo} 95% CI",
                alpha=0.2,
            )
        axes_hex[i % n_rows, i // n_rows].legend(fontsize=14, loc="upper right")
        axes_hex[i % n_rows, i // n_rows].set_xlabel("$c$")

    # Ensure a common density scale across all hexbins and add single colorbar at the far right
    if hb_list:
        global_max = max([hb.get_array().max() if hb.get_array().size > 0 else 0 for hb in hb_list])
        for hb in hb_list:
            hb.set_clim(0, global_max)
        try:
            fig_hex.colorbar(hb_list[-1], ax=axes_hex.ravel().tolist(), label="Density", location="right")
        except TypeError:
            # Some matplotlib versions don't accept 'location' or extra kwargs for Figure.colorbar
            fig_hex.colorbar(hb_list[-1], ax=axes_hex.ravel().tolist(), label="Density")

    fig_hex.supylabel("$e$")

    # Create filename based on plot_type and save into final_output_dir/figures/hexbin (no 'supplementary')
    filename = f"{dataset_key}_hex{'' if plot_type == 'test' else '_' + plot_type}.png"
    hex_dir = os.path.join(final_output_dir, "figures", "hexbin")
    os.makedirs(hex_dir, exist_ok=True)
    fig_hex.savefig(os.path.join(hex_dir, filename))
    plt.close(fig_hex)


def plot_hist(
    moo_heuristics: List[str],
    train_pareto_fronts: Dict[str, List[List]],
    n_cols: int,
    n_rows: int,
    title: str,
    final_output_dir: str,
    dataset_key: str,
) -> None:
    fig_hist, axes_hist = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharex=True, sharey=True, constrained_layout=True, squeeze=False
    )
    fig_hist.suptitle(title)
    for i, algo in enumerate(moo_heuristics):
        pf_lengths = [len(pf) for pf in train_pareto_fronts[algo]]
        if not pf_lengths:
            continue
        max_length = max(pf_lengths)
        axes_hist[i // n_cols, i % n_cols].hist(
            pf_lengths, bins=np.arange(1, max(33, max_length + 1)), align="right", rwidth=0.9
        )
        axes_hist[i // n_cols, i % n_cols].set_title(f"{algo}")
        axes_hist[i // n_cols, i % n_cols].set_xlabel(f"Cardinalities")
    fig_hist.supylabel("# of Pareto fronts")
    # Save into final_output_dir/figures/histograms (no 'supplementary')
    hist_dir = os.path.join(final_output_dir, "figures", "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    fig_hist.savefig(os.path.join(hist_dir, f"{dataset_key}_hist.png"))
    plt.close(fig_hist)


def plot_iterations_hv(
    res_var: pd.DataFrame, moo_heuristics: List[str], final_output_dir: str, dataset_key: str, title: str
) -> None:
    # Determine which heuristics have valid data for Iterations to Hypervolume
    valid_ithv_heuristics: List[str] = []
    # Ensure res_var is not None before proceeding
    if res_var is not None:
        for algo in moo_heuristics:
            algo_df = res_var.loc[res_var["Used_Representation"] == algo]
            iters = algo_df["metrics.sc_iterations"]
            # Condition for "ugly and redundant": all iteration values are the same
            if not iters.empty and iters.nunique() > 1:  # Check if not empty and has more than 1 unique value
                valid_ithv_heuristics.append(algo)
            else:
                print(f"Skipping Iterations to Hypervolume plot for {algo} due to constant/empty iteration count.")
    if not valid_ithv_heuristics:
        print(f"No valid data to plot Iterations to Hypervolume for problem: {dataset_key}. Skipping this plot.")
        return
    fig_ithv, axes_ithv = plt.subplots(
        1,
        len(valid_ithv_heuristics),
        figsize=(6 * len(valid_ithv_heuristics), 5),  # Adjust figsize dynamically
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    # Ensure axes_ithv is always an array, even if there's only one subplot
    if len(valid_ithv_heuristics) == 1:
        axes_ithv = [axes_ithv]
    fig_ithv.suptitle(title)
    for i, algo in enumerate(valid_ithv_heuristics):
        algo_df = res_var.loc[res_var["Used_Representation"] == algo]
        iters = algo_df["metrics.sc_iterations"]
        hv = algo_df["metrics.test_hypervolume"]
        algo_df_plot = pd.DataFrame({"Iterations": iters, "Hypervolume": hv})
        sns.scatterplot(data=algo_df_plot, x="Iterations", y="Hypervolume", ax=axes_ithv[i])
        # Fit regression line
        X = algo_df_plot["Iterations"].values.reshape(-1, 1)
        y = algo_df_plot["Hypervolume"].values
        # Check for NaNs and infs before fitting, and ensure X has enough unique points
        if not np.isnan(np.sum(X)) and not np.isinf(np.sum(X)) and len(np.unique(X)) > 1:
            reg = LinearRegression().fit(X, y)
            axes_ithv[i].plot(X, reg.predict(X), color="red", linewidth=2)
        else:
            print(f"Skipping regression line for {algo} due to insufficient data or constant X.")
        axes_ithv[i].set_title(f"{algo}")
    # Save into final_output_dir/figures/iterations_hv (no 'supplementary')
    ithv_dir = os.path.join(final_output_dir, "figures", "iterations_hv")
    os.makedirs(ithv_dir, exist_ok=True)
    fig_ithv.savefig(os.path.join(ithv_dir, f"{dataset_key}_ithv.png"))
    plt.close(fig_ithv)  # Close fig_ithv, not fig_kde


def compute_metric_dataframe(
    pareto_fronts: Dict[str, List[List]], metric: callable, name: str, *args, **kwargs
) -> pd.DataFrame:
    m_rows = []
    for algo in pareto_fronts.keys():
        for idx, front in enumerate(pareto_fronts[algo]):
            front_np = np.array(front)
            m = metric(front_np, *args, **kwargs)
            m_rows.append({"Used_Representation": algo, name: m})
    return pd.DataFrame(m_rows)


def compute_pareto_sacrifices_dataframes(pareto_fronts: Dict[str, List[List]], ref_name: str):
    c_rows = []
    e_rows = []
    ref_fronts = pareto_fronts[ref_name]
    for algo in pareto_fronts.keys():
        for idx, front in enumerate(pareto_fronts[algo]):
            front_np = np.array(front)
            ref_front_np = np.array(ref_fronts[idx])
            c_ps = c_pareto_sacrifice(front_np, ref_front_np)
            e_ps = e_pareto_sacrifice(front_np, ref_front_np)
            c_rows.append({"Used_Representation": algo, f"$c$-Pareto sacrifice": c_ps})
            e_rows.append({"Used_Representation": algo, f"$e$-Pareto sacrifice": e_ps})
    return pd.DataFrame(c_rows), pd.DataFrame(e_rows)


def plot_violin_metric(
    metric_df: pd.DataFrame,
    cfg: Dict[str, Any],
    problem: str,
    final_output_dir: str,
    name: str,
    allowed_algos: List[str] | None = None,
) -> None:
    # Filter to only MOO heuristics if provided
    if allowed_algos is not None:
        metric_df = metric_df[metric_df["Used_Representation"].isin(allowed_algos)]

    # Filter out Nan values from the metric column (turns out to be unnecessary)
    metric_df = metric_df[metric_df[name].notna()]
    # ================== Spread Violin Plot ==================
    fig_violin, ax_violin = plt.subplots(dpi=400)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.22)
    sns.violinplot(
        data=metric_df,
        x="Used_Representation",
        y=name,
        ax=ax_violin,
        inner="box",
        palette="tab10",
        hue="Used_Representation",
        legend=False,
    )
    ax_violin.set_title(f"{cfg['datasets'][problem]}", style="italic", fontsize=14)
    ax_violin.set_ylabel(name, fontsize=18, weight="bold")
    ax_violin.set_xlabel("")
    ax_violin.tick_params(axis="x", rotation=15)  #
    y_min = min(ax_violin.get_yticks())
    y_max = max(ax_violin.get_yticks())
    num_ticks = 7
    y_tick_positions = np.linspace(y_min, y_max, num_ticks)
    if y_min <= 0 <= y_max:
        # include zero explicitly
        y_tick_positions = np.unique(np.concatenate([y_tick_positions, [0.0]]))
    y_tick_positions = np.round(y_tick_positions, 3)
    ax_violin.set_ylim(y_min, y_max)
    ax_violin.set_yticks(y_tick_positions)
    ax_violin.set_yticklabels([f"{x:.3g}" for x in y_tick_positions])
    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.tight_layout()
    # Save into final_output_dir/figures/violins (sanitize name) — removed 'supplementary'
    violin_dir = os.path.join(final_output_dir, "figures", "violins")
    os.makedirs(violin_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-_\. ]", "", name).replace(" ", "_")
    fig_violin.savefig(os.path.join(violin_dir, f"{datasets_map[problem]}_violin_{safe_name}.png"))
    plt.close(fig_violin)


def plot_swarm_box_metric(
    metric_df: pd.DataFrame,
    cfg: Dict[str, Any],
    problem: str,
    final_output_dir: str,
    name: str,
    allowed_algos: List[str] | None = None,
) -> None:
    """
    Boxplot (summary) with overlaid swarmplot (raw points) per algorithm.
    Saves to final_output_dir/figures/swarm.
    """
    # Filter to only MOO heuristics if provided
    if allowed_algos is not None:
        metric_df = metric_df[metric_df["Used_Representation"].isin(allowed_algos)]

    # Drop NaNs in the plotted column
    metric_df = metric_df[metric_df[name].notna()]

    fig_swarm, ax_swarm = plt.subplots(dpi=400)
    plt.subplots_adjust(left=0.2, right=0.95, top=0.92, bottom=0.22)

    # Draw boxplot first (transparent face; summary under swarm points)
    sns.boxplot(
        data=metric_df,
        x="Used_Representation",
        y=name,
        ax=ax_swarm,
        width=0.5,
        showfliers=False,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    # Overlay swarmplot
    sns.swarmplot(
        data=metric_df,
        x="Used_Representation",
        y=name,
        ax=ax_swarm,
        hue="Used_Representation",
        palette="tab10",
        dodge=False,
        size=3,
        alpha=0.8,
        legend=False,
    )

    ax_swarm.set_title(f"{cfg['datasets'][problem]}", style="italic", fontsize=14)
    ax_swarm.set_ylabel(name, fontsize=18, weight="bold")
    ax_swarm.set_xlabel("")
    ax_swarm.tick_params(axis="x", rotation=15)

    # Determine y-range from the actual data when available
    y_vals = metric_df[name].dropna().to_numpy() if name in metric_df.columns else np.array([])
    if y_vals.size > 0:
        y_vals = y_vals[np.isfinite(y_vals)]
    if y_vals.size == 0:
        y_min = min(ax_swarm.get_yticks())
        y_max = max(ax_swarm.get_yticks())
    else:
        y_min = float(np.min(y_vals))
        y_max = float(np.max(y_vals))

    # Add a small padding unless the range is zero
    pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.5
    y_min -= pad
    y_max += pad

    num_ticks = 7

    # Detect integer-like data (within rounding tolerance)
    all_integer = False
    if y_vals.size > 0:
        all_integer = bool(np.all(np.isclose(y_vals, np.round(y_vals))))

    # Build tick positions, always ensuring equal spacing and including 0 if it's in the data range
    if all_integer:
        y_min_int = int(np.floor(y_min))
        y_max_int = int(np.ceil(y_max))
        span = max(1, y_max_int - y_min_int)
        n_ticks = min(num_ticks, span + 1)
        step = max(1, int(np.ceil(span / (n_ticks - 1))))
        y_tick_positions = np.arange(y_min_int, y_max_int + 1, step)
        # If 0 lies in the original data range but isn't in ticks, recompute evenly spaced integer ticks that include 0
        if y_vals.size > 0 and np.min(y_vals) <= 0 <= np.max(y_vals) and 0 not in y_tick_positions:
            y_tick_positions = np.linspace(y_min_int, y_max_int, n_ticks, dtype=int)
    else:
        y_tick_positions = np.linspace(y_min, y_max, num_ticks)
        if y_vals.size > 0 and np.min(y_vals) <= 0 <= np.max(y_vals) and 0 not in y_tick_positions:
            y_tick_positions = np.linspace(min(y_min, 0.0), max(y_max, 0.0), num_ticks)

    y_tick_positions = np.unique(np.round(y_tick_positions, 6))

    ax_swarm.set_ylim(y_min, y_max)
    ax_swarm.set_yticks(y_tick_positions)
    if all_integer:
        ax_swarm.set_yticklabels([str(int(x)) for x in y_tick_positions])
    else:
        ax_swarm.set_yticklabels([f"{x:.3g}" for x in y_tick_positions])

    plt.xticks(rotation=15, ha="right", fontsize=12)
    plt.tight_layout()

    swarm_dir = os.path.join(final_output_dir, "figures", "swarm")
    os.makedirs(swarm_dir, exist_ok=True)
    safe_name = re.sub(r"[^\w\-_\. ]", "", name).replace(" ", "_")
    fig_swarm.savefig(os.path.join(swarm_dir, f"{datasets_map[problem]}_swarm_{safe_name}.png"))
    plt.close(fig_swarm)


def prepare_reference_stats(
    pareto_solutions: Dict[str, np.ndarray], cfg: Dict[str, Any]
) -> Tuple[Dict[str, Tuple[np.ndarray, Tuple[str, str]]], Dict[str, Tuple[np.ndarray, Tuple[str, str]]]]:
    """
    Prepares mean points and covariance (for confidence ellipses) of reference (SOO) heuristics.
    Extracts soo_pareto_fronts
    Uses cfg['reference_heuristics'] (raw_name -> display_name).
    """
    ref_cfg = cfg.get("reference_heuristics", {})
    averages: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    covs: Dict[str, Tuple[np.ndarray, Tuple[str, str]]] = {}
    for raw_name, display_name in ref_cfg.items():
        style = ga_baselines.get(raw_name, ("o", "black"))
        if display_name in pareto_solutions and len(pareto_solutions[display_name]) > 0:
            data = pareto_solutions[display_name]
            averages[display_name] = (np.mean(data, axis=0), style)
            # Need at least 2 points for covariance
            if data.shape[0] > 1:
                covs[display_name] = (np.cov(data.T), style)
            else:
                # Fallback: zero covariance
                covs[display_name] = (np.zeros((2, 2)), style)
    return averages, covs


def compute_nan_percentage_per_algo(
    metric_df: pd.DataFrame, value_col: str, algo_list: List[str]
) -> Dict[str, Optional[float]]:
    """
    Returns a dict algo -> percentage of NaNs in value_col (0..100) or None if no rows for algo.
    """
    out: Dict[str, Optional[float]] = {}
    if metric_df is None or metric_df.empty or value_col not in metric_df.columns:
        for a in algo_list:
            out[a] = None
        return out
    for algo in algo_list:
        sub = metric_df[metric_df["Used_Representation"] == algo]
        total = int(sub.shape[0])
        if total == 0:
            out[algo] = None
            continue
        n_nans = int(sub[value_col].isna().sum())
        out[algo] = 100.0 * n_nans / float(total)
    return out


def plot_sampled_pareto_with_refs(
    algo_name: str,
    pareto_fronts: Dict[str, List[List]],
    reference_fronts: Dict[str, List[List]],
    ref_display_name: str,
    dataset_title: str,
    final_output_dir: str,
    dataset_key: str,
    plot_type: str = "test",
) -> None:
    """
    Plot 8 sampled Pareto fronts (indices: 0, 7, 15, 23, 31, 39, 47, 55) for a single algorithm and
    overlay the corresponding GA reference elitist (single point) in the same color with a different marker.

    Saves to: final_output_dir/figures/pareto_samples
    """
    # Guard: ensure data exists
    fronts = pareto_fronts.get(algo_name, [])
    ref_fronts_list = reference_fronts.get(ref_display_name, [])
    if not fronts or not ref_fronts_list:
        return

    sample_indices = [0, 7, 15, 23, 31, 39, 47, 55]
    # Bound indices by available fronts
    sample_indices = [i for i in sample_indices if i < len(fronts) and i < len(ref_fronts_list)]
    if len(sample_indices) == 0:
        return

    # Prepare figure (2x4 grid)
    n_samples = len(sample_indices)
    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle(f"{dataset_title} — {algo_name} ({plot_type.capitalize()})")

    # Color cycle
    palette = sns.color_palette("tab10", n_samples)

    for k, idx in enumerate(sample_indices):
        ax = axes[k // n_cols, k % n_cols]
        color = palette[k % len(palette)]

        # Extract current Pareto front and corresponding GA reference front
        pf = np.array(fronts[idx])
        ref_pf = np.array(ref_fronts_list[idx])

        # Plot Pareto front as line/scatter in color
        if pf.size > 0:
            ax.plot(pf[:, 0], pf[:, 1], "-", color=color, linewidth=1.5, label=f"PF {k + 1}")
            ax.scatter(pf[:, 0], pf[:, 1], s=12, color=color)

        # Plot single GA elitist point (first item of reference PF) with distinct marker in same color
        if ref_pf.size > 0:
            elitist = ref_pf[0]  # assume first item is the elitist point
            ax.scatter(elitist[0], elitist[1], s=50, marker="X", color=color, edgecolor="black", linewidth=0.8, label=f"SOO Elitist {k + 1}")

        ax.set_title(f"Seed $s_{k + 1}$")
        ax.set_xlabel("$c$")
        ax.set_ylabel("$e$")

        # Bound axes to [0,1] if data suggests that domain (light guard)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)

        # Add per-subplot legend (show PF and GA for this run)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=14, loc="upper right")

    # Save
    out_dir = os.path.join(final_output_dir, "figures", "pareto_samples")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{dataset_key}_{algo_name.replace(' ', '_')}_samples_{plot_type}.png"
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname))
    plt.close(fig)


def create_plots():
    configure_style()
    config = load_config()
    final_output_dir = f"{config['output_directory']}"

    tuning_info: Dict[str, Dict[str, Dict]] = {}

    # Merge MOO and reference heuristics for data loading
    ref_heurs = config.get("reference_heuristics", {})
    all_heuristics: Dict[str, str] = {**config["heuristics"], **ref_heurs}

    # Accumulator for Pareto Sacrifice undefined table
    pareto_sacrifice_nan_stats: Dict[str, Dict[str, Optional[float]]] = {}

    for problem in config["datasets"]:
        counter = 0
        first = True
        train_pareto_fronts: Dict[str, List[List]] = {}
        train_pareto_solutions: Dict[str, List[List[float]]] = {}
        test_pareto_fronts: Dict[str, List[List]] = {}
        test_pareto_solutions: Dict[str, List[List[float]]] = {}
        res_var = None
        tuning_info[problem] = {}

        # Load runs for all heuristics (MOO and reference)
        for heuristic, renamed_heuristic in all_heuristics.items():
            train_pareto_fronts[renamed_heuristic] = []
            train_pareto_solutions[renamed_heuristic] = []
            test_pareto_fronts[renamed_heuristic] = []
            test_pareto_solutions[renamed_heuristic] = []

            fold_df, root_df = load_fold_dataframe(heuristic, problem, config)
            if root_df is not None and not root_df.empty:
                string_dict = root_df["params.tuned_params"].iloc[0]
                string_dict = re.sub(r"([A-Za-z_]\w*)\([^)]*\)", r"'\1'", string_dict)
                try:
                    tuning_info[problem][heuristic] = ast.literal_eval(string_dict)
                except Exception:
                    tuning_info[problem][heuristic] = {}
                tuning_path = root_df["artifact_uri"].iloc[0].split("suprb-experimentation/")[-1]
                tuning_path = os.path.join(tuning_path, "param_history.json")
                if os.path.exists(tuning_path):
                    with open(tuning_path, "r") as f:
                        param_history = json.load(f)
                        if len(param_history.keys()) > 0:
                            n_calls = len(param_history[list(param_history.keys())[0]])
                            tuning_info[problem][heuristic]["$n_{trials}$"] = n_calls

            if fold_df is not None:
                counter += 1
                name_col = [renamed_heuristic] * fold_df.shape[0]
                current_res = fold_df.assign(Used_Representation=name_col)
                if first:
                    first = False
                    res_var = current_res
                else:
                    res_var = pd.concat([res_var, current_res])
                process_artifact_paths(
                    current_res,
                    renamed_heuristic,
                    train_pareto_fronts,
                    train_pareto_solutions,
                    test_pareto_fronts,
                    test_pareto_solutions,
                )

        if counter == 0:
            print(f"No data for problem {problem}, skipping.")
            continue

        # Convert collected solution lists into arrays
        for key in train_pareto_solutions.keys():
            train_pareto_solutions[key] = np.array(train_pareto_solutions[key])
            test_pareto_solutions[key] = np.array(test_pareto_solutions[key])

        # MOO heuristics now are exactly the display names of config['heuristics']
        moo_heuristics = list(config["heuristics"].values())

        # Prepare reference (SOO) overlays
        test_ref_averages, test_ref_std = prepare_reference_stats(test_pareto_solutions, config)
        train_ref_averages, train_ref_std = prepare_reference_stats(train_pareto_solutions, config)

        n_algs = len(moo_heuristics)
        n_cols, n_rows = determine_layout(n_algs)
        dataset_key = datasets_map[problem]
        dataset_title = config["datasets"][problem] + " (" +  dataset_key.upper() + ")"

        # Plots
        plot_hexbin(
            moo_heuristics,
            test_pareto_solutions,
            test_ref_averages,
            test_ref_std,
            n_cols,
            n_rows,
            dataset_title,
            final_output_dir,
            dataset_key,
            plot_type="test",
        )
        plot_hexbin(
            moo_heuristics,
            train_pareto_solutions,
            train_ref_averages,
            train_ref_std,
            n_cols,
            n_rows,
            dataset_title,
            final_output_dir,
            dataset_key,
            plot_type="train",
        )

        # New: sampled Pareto fronts with GA reference elitists (same color, different marker)
        if len(config["reference_heuristics"]) > 0:
            ref_display_name = config["reference_heuristics"][list(config["reference_heuristics"].keys())[0]]
            for algo in moo_heuristics:
                plot_sampled_pareto_with_refs(
                    algo,
                    test_pareto_fronts,
                    test_pareto_fronts,  # GA reference fronts are stored under reference_heuristics display name
                    ref_display_name,
                    dataset_title,
                    final_output_dir,
                    dataset_key,
                    plot_type="test",
                )
                plot_sampled_pareto_with_refs(
                    algo,
                    train_pareto_fronts,
                    train_pareto_fronts,
                    ref_display_name,
                    dataset_title,
                    final_output_dir,
                    dataset_key,
                    plot_type="train",
                )

        plot_hist(moo_heuristics, train_pareto_fronts, n_cols, n_rows, dataset_title, final_output_dir, dataset_key)
        plot_iterations_hv(res_var, moo_heuristics, final_output_dir, dataset_key, dataset_title)

        moo_heuristics = list(config["heuristics"].values())

        # Metrics (filtered to only MOO heuristics in plots)
        spread_df = compute_metric_dataframe(train_pareto_fronts, metric_spread, "Spread")
        plot_violin_metric(spread_df, config, problem, final_output_dir, "Spread", allowed_algos=moo_heuristics)
        plot_swarm_box_metric(spread_df, config, problem, final_output_dir, "Spread", allowed_algos=moo_heuristics)

        hv_df = compute_metric_dataframe(
            test_pareto_fronts, metric_hypervolume, "Test Hypervolume", reference_point=np.array([1.0, 1.0])
        )
        plot_violin_metric(hv_df, config, problem, final_output_dir, "Test Hypervolume", allowed_algos=moo_heuristics)
        plot_swarm_box_metric(hv_df, config, problem, final_output_dir, "Test Hypervolume", allowed_algos=moo_heuristics)

        if len(config["reference_heuristics"]) > 0:
            reference_heuristic = config["reference_heuristics"][list(config["reference_heuristics"].keys())[0]]
            c_ps, e_ps = compute_pareto_sacrifices_dataframes(test_pareto_fronts, reference_heuristic)


            plot_violin_metric(
                c_ps,
                config,
                problem,
                final_output_dir,
                f"$c$-Pareto sacrifice",
                allowed_algos=moo_heuristics,
            )
            plot_swarm_box_metric(
                c_ps,
                config,
                problem,
                final_output_dir,
                f"$c$-Pareto sacrifice",
                allowed_algos=moo_heuristics,
            )

            plot_violin_metric(
                e_ps,
                config,
                problem,
                final_output_dir,
                f"$e$-Pareto sacrifice",
                allowed_algos=moo_heuristics,
            )
            plot_swarm_box_metric(
                e_ps,
                config,
                problem,
                final_output_dir,
                f"$e$-Pareto sacrifice",
                allowed_algos=moo_heuristics,
            )

            # ---- Collect NaN percentage for Pareto Sacrifice undefined table ----
            sacrifice_col = f"$c$-Pareto sacrifice"
            per_algo_nan = compute_nan_percentage_per_algo(
                c_ps[c_ps["Used_Representation"].isin(moo_heuristics)],
                sacrifice_col,
                moo_heuristics,
            )
            pareto_sacrifice_nan_stats[problem] = per_algo_nan

    # After all datasets processed: emit LaTeX table
    if len(pareto_sacrifice_nan_stats) > 0:
        generate_pareto_sacrifice_undefined_table(pareto_sacrifice_nan_stats, config, moo_heuristics, final_output_dir)
    generate_tuning_tables(tuning_info, config, final_output_dir)


if __name__ == "__main__":
    create_plots()
