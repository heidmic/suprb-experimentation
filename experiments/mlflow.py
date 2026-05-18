import json
import warnings
from typing import Optional

import mlflow
import numpy as np
from sklearn.base import BaseEstimator
from suprb import SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.multi_objective import MOLogger

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
        try:
            mlflow.log_dict(d, name)
        except TypeError as e1:
            try:
                mlflow.log_text(json.dumps(d, default=str), name)
            except TypeError as e2:
                warnings.warn(f"Logging of {name} as json has failed twice with reasons: {e1}; {e2}")
                print(d)
    except:
        warnings.warn(f"Logging of {name} as json has failed with Error (other than TypeErrors)")
        print(d)


def log_experiment(experiment: Experiment):
    _log_experiment(experiment, parent_name="", depth=0)


def _log_experiment(experiment: Experiment, parent_name: str, depth: int) -> dict:
    run_name = f"{parent_name}/{experiment.name}"
    with mlflow.start_run(run_name=run_name, nested=depth > 0) as parent_run:

        # Set root tag
        if depth == 0:
            mlflow.set_tag("root", True)

        # Log tuning results
        if hasattr(experiment, "tuned_params_"):
            log_tuning(experiment)

        if experiment.experiments:
            # Call log on nested experiments
            results = []
            for subexperiment in experiment.experiments:
                result = _log_experiment(subexperiment, parent_name=run_name, depth=depth + 1)
                if result is not None:
                    results.append(result)

            with mlflow.start_run(run_name=f"{run_name}.averaged_exp", nested=True) as average_run:
                average_results = {key: np.mean(value) for key, value in _expand_list(results).items()}
                log_run_result(average_results)

        else:
            # Check if experiment has evaluations
            if hasattr(experiment, "results_") and hasattr(experiment, "estimators_"):
                # Log cv folds
                for i, (estimator, result) in enumerate(
                    zip(experiment.estimators_, _expand_dict(experiment.results_)), 1
                ):
                    with mlflow.start_run(
                        run_name=f"{run_name}.fold-{i}/{len(experiment.estimators_)}", nested=True
                    ) as cv_run:
                        log_run(estimator)
                        log_run_result(result)

                        mlflow.set_tag("fold", True)

                # Log cv average
                with mlflow.start_run(run_name=f"{run_name}.averaged_cv", nested=True) as average_run:
                    average_results = {
                        key: np.mean(value) if not isinstance(value, list) else 0
                        for key, value in experiment.results_.items()
                    }
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

    try_log_dict(history, "param_history.json")

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
    # Log model parameters
    try_log_dict(estimator.get_params(), "params.json")

    logger = _get_default_logger(estimator)
    if logger is not None:
        # Log fitting metrics
        for key, values in logger.metrics_.items():
            for step, value in values.items():
                mlflow.log_metric(key=key, value=value, step=step)

        if isinstance(logger, MOLogger):
            try_log_dict(logger.pareto_fronts_, "pareto_fronts.json")


def log_run_result(result: dict):
    if "test_pf_fitness" in result:
        test_pf_fitness = result.pop("test_pf_fitness")
        try_log_dict({"test_pf_fitness": test_pf_fitness}, "test_pareto_front.json")
    mlflow.log_metrics(result)
