import json
import warnings
from typing import Optional

import mlflow
import numpy as np
from sklearn.base import BaseEstimator
from suprb import SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger

from experiments import Experiment


def _expand_dict(d: dict) -> list:
    """Convert a dict of lists into a list of dicts."""
    return [dict(zip(d, t)) for t in zip(*d.values())]


def _expand_list(l: list) -> dict:
    """Convert a list of dicts into a dict of lists."""
    keys = set().union(*l)
    return {k: [dic[k] if k in dic else None for dic in l] for k in keys}


def _get_default_logger(estimator: BaseEstimator) -> Optional[DefaultLogger]:
    if isinstance(estimator, SupRB):
        logger = estimator.logger_
        if isinstance(logger, DefaultLogger):
            return logger
        elif isinstance(logger, CombinedLogger):
            for name, sublogger in logger.loggers_:
                if isinstance(sublogger, DefaultLogger):
                    return sublogger


def try_log_dict(d: dict, name: str):
    try:
        mlflow.log_dict(d, name)
    except TypeError as e1:
        try:
            mlflow.log_text(json.dumps(d, default=str), name)
        except TypeError as e2:
            warnings.warn(f"Logging of {name} as json has failed twice with reasons: {e1}; {e2}")
            print(d)


def log_experiment(experiment: Experiment):
    _log_experiment(experiment, parent_name='', depth=0)


def _log_experiment(experiment: Experiment, parent_name: str, depth: int) -> dict:
    run_name = f"{parent_name}/{experiment.name}"
    with mlflow.start_run(run_name=run_name, nested=depth > 0) as parent_run:

        # Set root tag
        if depth == 0:
            mlflow.set_tag("root", True)

        # Log tuning results
        if hasattr(experiment, 'tuned_params_'):
            log_tuning(experiment)

        if experiment.experiments:
            # Call log on nested experiments
            results = []
            for subexperiment in experiment.experiments:
                result = _log_experiment(subexperiment, parent_name=run_name, depth=depth + 1)
                if result is not None:
                    results.append(result)
                if "Decisioasdn Tree" in experiment.name:
                    for i, (estimator, result) in enumerate(
                            zip(subexperiment.estimators_, _expand_dict(subexperiment.results_)),
                            1):
                        n_nodes = estimator.tree_.node_count
                        children_left = estimator.tree_.children_left
                        children_right = estimator.tree_.children_right
                        feature = estimator.tree_.feature
                        threshold = estimator.tree_.threshold

                        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
                        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
                        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
                        while len(stack) > 0:
                            # `pop` ensures each node is only visited once
                            node_id, depth = stack.pop()
                            node_depth[node_id] = depth

                            # If the left and right child of a node is not the same we have a split
                            # node
                            is_split_node = children_left[node_id] != children_right[node_id]
                            # If a split node, append left and right children and depth to `stack`
                            # so we can loop through them
                            if is_split_node:
                                stack.append((children_left[node_id], depth + 1))
                                stack.append((children_right[node_id], depth + 1))
                            else:
                                is_leaves[node_id] = True

                        mlflow.log_metric("elitist_complexity", sum(is_leaves))

            with mlflow.start_run(run_name=f"{run_name}.averaged_exp", nested=True) as average_run:
                average_results = {key: np.mean(value) for key, value in _expand_list(results).items()}
                log_run_result(average_results)

        else:
            # Check if experiment has evaluations
            if hasattr(experiment, 'results_') and hasattr(experiment, 'estimators_'):
                # Log cv folds
                for i, (estimator, result) in enumerate(zip(experiment.estimators_, _expand_dict(experiment.results_)),
                                                        1):
                    with mlflow.start_run(run_name=f"{run_name}.fold-{i}/{len(experiment.estimators_)}",
                                          nested=True) as cv_run:
                        log_run(estimator)
                        log_run_result(result)

                        mlflow.set_tag('fold', True)

                        if "Decisioasdn Tree" in experiment.name:
                            n_nodes = estimator.tree_.node_count
                            children_left = estimator.tree_.children_left
                            children_right = estimator.tree_.children_right
                            feature = estimator.tree_.feature
                            threshold = estimator.tree_.threshold

                            node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
                            is_leaves = np.zeros(shape=n_nodes, dtype=bool)
                            stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
                            while len(stack) > 0:
                                # `pop` ensures each node is only visited once
                                node_id, depth = stack.pop()
                                node_depth[node_id] = depth

                                # If the left and right child of a node is not the same we have a split
                                # node
                                is_split_node = children_left[node_id] != children_right[node_id]
                                # If a split node, append left and right children and depth to `stack`
                                # so we can loop through them
                                if is_split_node:
                                    stack.append((children_left[node_id], depth + 1))
                                    stack.append((children_right[node_id], depth + 1))
                                else:
                                    is_leaves[node_id] = True

                            mlflow.log_metric("elitist_complexity", sum(is_leaves))

                # Log cv average
                with mlflow.start_run(run_name=f"{run_name}.averaged_cv", nested=True) as average_run:
                    average_results = {key: np.mean(value) for key, value in experiment.results_.items()}
                    log_run_result(average_results)
            else:
                average_results = None

            # Set leaf tag
            mlflow.set_tag("leaf", True)

        return average_results


def log_tuning(experiment: Experiment):
    tuner = experiment.tuner
    tuning_result = experiment.tuning_results_

    # Log params
    mlflow.log_params({f"tuning_{key}": value for key, value in tuner.get_params().items()})
    mlflow.log_param("parameter_space", experiment.param_space)

    # Log history
    for step, objective_value in enumerate(tuning_result.objective_history):
        mlflow.log_metric("objective_function", objective_value, step=step)

    history = _expand_list(tuning_result.params_history)

    try_log_dict(history, 'param_history.json')

    # Try to log floaty params as metrics. `mlflow.log_metric` only accepts float values.
    for step, params in enumerate(tuning_result.params_history):
        valid_params = {}

        for key, value in params.items():
            try:
                valid_params[key] = float(value)
            except ValueError:
                pass

        mlflow.log_metrics(valid_params, step=step)

    # Final tuning result
    mlflow.log_param("tuned_params", experiment.tuned_params_)


def log_run(estimator: BaseEstimator):
    logger = _get_default_logger(estimator)

    try_log_dict(estimator.get_params(), 'params.json')
    try:
        try_log_dict(logger.get_elitist(estimator), 'elitist.json')
    except AttributeError:
        print("Elitist not logged!")

    if logger is not None:
        # Log fitting metrics
        for key, values in logger.metrics_.items():
            for step, value in values.items():
                mlflow.log_metric(key=key, value=value, step=step)

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print("blyat", sum(is_leaves), n_nodes)
    mlflow.log_metric("elitist_complexity", sum(is_leaves))


def log_run_result(result: dict):
    mlflow.log_metrics(result)
