import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from evaluation.utils import tex_escape


def plot_metric(
        metrics: pd.DataFrame,
        x: str = 'elitist_fitness',
        y: str = 'optimizer',
        xlabel: str = r'$F$',
        ylabel: str = r'Metaheuristic',
        ax=None,
        **kwargs):
    metrics = metrics.reset_index().rename(columns=tex_escape)

    params = dict(cut=0, palette='colorblind', scale='width') | kwargs
    ax = sns.violinplot(data=metrics, y=tex_escape(y), x=tex_escape(x), ax=ax, **params)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax



