"""
Rule-pool-size-over-iterations plot.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from mlflow.tracking import MlflowClient

from condition_csv_loader import get_condition_df


def plot_pool_size_growth(
    condition_heuristics: dict, problem: str, final_output_dir: str, dataset_key: str, subdir: str, tracking_uri: str = None
) -> None:
    """
    condition_heuristics: dict mapping condition tag -> display name, e.g.
        {"es_nopruning_noadaptive": "ES Baseline", "es_pruning_noadaptive": "ES + Pruning", ...}
    subdir: the mlruns_csv subdirectory for this setting (e.g. "FINAL_EVAL_ES"),
        same value as cfg["data_directory"].split("/")[-1].
    """
    client = MlflowClient(tracking_uri=tracking_uri)
    fig, ax = plt.subplots(dpi=400)

    any_data = False
    for condition, display_name in condition_heuristics.items():
        try:
            df = get_condition_df(problem, condition, subdir)
        except (FileNotFoundError, ValueError) as e:
            print(f"[pool_size] {e} -- skipping this line.")
            continue

        if df.empty or "run_id" not in df.columns:
            print(f"[pool_size] no run_id data for condition={condition}, problem={problem} -- skipping this line.")
            continue

        all_series = []
        for run_id in df["run_id"]:
            hist = client.get_metric_history(run_id, "pool_size")
            if hist:
                all_series.append([(h.step, h.value) for h in hist])

        if not all_series:
            print(f"[pool_size] no pool_size metric found for condition={condition}, problem={problem} -- skipping this line.")
            continue
        any_data = True

        max_len = max(len(s) for s in all_series)
        arr = np.full((len(all_series), max_len), np.nan)
        for i, series in enumerate(all_series):
            for step, val in series:
                if step < max_len:
                    arr[i, step] = val

        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        x = np.arange(max_len)
        ax.plot(x, mean, label=display_name, linewidth=1.5)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15)

    if not any_data:
        print(f"[pool_size] no data at all for problem={problem} -- skipping plot entirely.")
        plt.close(fig)
        return

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Rule pool size")
    ax.set_title(dataset_key, style="italic", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    out_dir = os.path.join(final_output_dir, "figures", "pool_size")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{dataset_key}_pool_size.png"))
    plt.close(fig)
