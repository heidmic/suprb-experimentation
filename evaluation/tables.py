import pandas as pd

from evaluation.utils import concise_metrics_summary, latex_props


def format_summary_table(metrics: pd.DataFrame):
    summary = concise_metrics_summary(metrics)
    # summary.rename(columns=, inplace=True)
    s = summary.style
    s = s.highlight_max(subset=(slice(None), (['elitist_fitness', 'train_r2', 'test_r2'], ['mean', 'median'])), props=latex_props(('bold',)))
    s = s.highlight_min(subset=(slice(None), (['elitist_complexity', 'train_mean_squared_error', 'test_mean_squared_error'], ['mean', 'median'])),props=latex_props(('bold',)))
    s = s.format(precision=3)
    print(s.to_latex(multicol_align='c'))