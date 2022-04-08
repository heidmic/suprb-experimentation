from __future__ import annotations

import re

import mlflow.tracking
import pandas as pd

OPTIMIZERS = ('RS', 'GA', 'ACO', 'GWO', 'PSO', 'ABC')

PROBLEMS = {
    'combined_cycle_power_plant': 'CCPP',
    'airfoil_self_noise': 'ASN',
    'concrete_strength': 'CS',
    'energy_cool': 'EEC',
}

PRETTY_COLUMNS = {
    'elitist_fitness': r'$F$',
    'elitist_complexity': r'$C$',
    'test_r2': r'R^2_{\text{test}}',
    'train_r2': r'R^2_{\text{train}}',
    'test_mean_squared_error': r'\text{MSE}_{\text{test}}',
    'train_mean_squared_error': r'\text{MSE}_{\text{train}}',
}


def tex_escape(text: str) -> str:
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def latex_props(formats: list[str]) -> str:
    f = {'bold': 'textbf', 'italics': 'emph', 'underline': 'underline', }
    return ';'.join([f"{fmt}:--rwrap" for fmt in {value for key, value in f.items() if key in formats}])


def get_folds_with_name(problem: str, optimizer: str) -> pd.DataFrame:
    return mlflow.search_runs([mlflow.get_experiment_by_name(problem).experiment_id],
                              f"tags.`mlflow.runName` like '/{optimizer.upper()} Evaluation/%' and tags.fold = 'True'")


def get_metrics(problem: str, optimizer: str, columns: list, rename: dict = None) -> pd.DataFrame:
    runs = get_folds_with_name(problem=problem, optimizer=optimizer)
    metrics = runs[columns]
    metrics = metrics.rename(columns=lambda x: x.split('.')[1])
    metrics.rename(columns=rename, inplace=True)
    if 'test_mean_squared_error' in metrics.columns:
        metrics['test_mean_squared_error'] *= -1
    metrics.index.name = 'run'
    return metrics


def get_relevant_metrics(problem: str, optimizer: str) -> pd.DataFrame:
    columns = ['metrics.test_r2', 'metrics.training_score', 'metrics.test_neg_mean_squared_error',
               'metrics.elitist_error', 'metrics.elitist_complexity', 'metrics.elitist_fitness']
    rename = {'training_score': 'train_r2', 'elitist_error': 'train_mean_squared_error',
              'test_neg_mean_squared_error': 'test_mean_squared_error'}
    return get_metrics(problem=problem, optimizer=optimizer, columns=columns, rename=rename)


def get_relevant_metrics_for_optimizers(problem: str, optimizers=OPTIMIZERS) -> pd.DataFrame:
    metrics = pd.concat({optimizer: get_relevant_metrics(problem=problem, optimizer=optimizer)
                         for optimizer in optimizers})
    metrics.index.names = ['optimizer', 'run']
    return metrics


def get_relevant_metrics_for_all_optimizers_and_problems():
    metrics = pd.concat({problem: get_relevant_metrics_for_optimizers(problem) for problem in PROBLEMS})
    metrics.index = metrics.index.rename('problem', level=0)
    return metrics


def metrics_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(by='optimizer').describe().drop(columns='count', level=1)


def concise_metrics_summary(metrics: pd.DataFrame, include_median=False) -> pd.DataFrame:
    columns = ['mean', 'std']
    if include_median:
        columns.append('50%')
    metrics = metrics.groupby(by='optimizer').describe().loc[:, (slice(None), columns)]
    if include_median:
        metrics.rename(columns={'50%': 'median'}, level=1, inplace=True)
    return metrics


def get_metric_history(problem: str, optimizer: str, column: str) -> pd.Series:
    runs = get_folds_with_name(problem, optimizer)
    client = mlflow.tracking.MlflowClient()

    history = runs.run_id.map(lambda run_id: pd.Series({metric.step: metric.value
                                                        for metric in client.get_metric_history(run_id, column)}))

    metrics_history = pd.DataFrame(history.to_list()).stack()
    metrics_history.index.names = ['run', 'it']
    metrics_history.name = column

    return metrics_history


def get_metric_history_for_optimizers(problem: str, column: str,
                                      optimizers=('RS', 'GA', 'ACO', 'GWO', 'PSO', 'ABC')) -> pd.Series:
    history = pd.concat({optimizer: get_metric_history(problem, optimizer, column) for optimizer in optimizers})
    history.index = history.index.rename('optimizer', level=0)
    return history


def get_metric_history_for_all_optimizers_and_problems(column: str) -> pd.Series:
    histories = pd.concat({problem: get_metric_history_for_optimizers(problem, column) for problem in PROBLEMS})
    histories.index = histories.index.rename('problem', level=0)
    return histories
