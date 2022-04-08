import pandas as pd
import seaborn as sns

from evaluation.utils import tex_escape

sns.set_theme(font_scale=1.35, style='whitegrid', palette='colorblind', font='serif')


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


def plot_metric_history(history: pd.Series, x='it', y='elitist_fitness'):
    history = history.reset_index().rename(columns=tex_escape)

    grid = sns.relplot(data=history, col='problem', x=tex_escape(x), y=tex_escape(y), hue='optimizer', kind='line',
                       markers=True,
                       style='optimizer', col_wrap=2)

    return grid
