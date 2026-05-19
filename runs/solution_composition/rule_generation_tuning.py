import math

import click
import mlflow
from optuna import Trial
from sklearn.utils import Bunch, shuffle

from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from runs.solution_composition.shared_config import (
    shared_tuning_params,
    load_dataset,
    global_params,
    dataset_params,
    random_state,
)


@click.command()
@click.option("-p", "--problem", type=click.STRING, default="airfoil_self_noise")
def run(problem: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    @param_space("rule_discovery")
    def rule_discovery_space(trial: Trial, params: Bunch):
        sigma_space = [0, math.sqrt(X.shape[1])]

        # params.mutation = trial.suggest_categorical('mutation', ['HalfnormIncrease', 'Normal'])
        # params.mutation = getattr(es.mutation, params.mutation)()
        params.mutation__sigma = trial.suggest_float("mutation__sigma", *sigma_space)
        params.delay = trial.suggest_int("delay", 1, 200)
        params.init__fitness__alpha = trial.suggest_float("init__fitness__alpha", 0.05, 1)

        # if isinstance(params.mutation, es.mutation.HalfnormIncrease):
        #     params.init = rule.initialization.MeanInit()
        # else:
        #     params.init = rule.initialization.HalfnormInit()
        #     params.init__sigma = trial.suggest_float('init__sigma', *sigma_space)

    params = global_params | dataset_params.get(problem, {})

    experiment = Experiment(name=f"{problem} RG Tuning", params=params, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **shared_tuning_params)
    experiment.with_tuning(rule_discovery_space, tuner=tuner)

    experiment.perform(evaluation=None)

    mlflow.set_experiment("RG Tuning")
    log_experiment(experiment)


if __name__ == "__main__":
    run()
