import math

import click
import mlflow
import numpy as np
from optuna import Trial
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from suprb2 import rule
from suprb2.optimizer.rule import es

from experiments import Experiment
from experiments.evaluation import CrossValidateTest
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from runs.individual_optimizers.shared_config import shared_tuning_params, load_dataset, global_params, dataset_params, \
    estimator, random_state


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
def run(problem: str):
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    @param_space('rule_generation')
    def rule_generation_space(trial: Trial, params: Bunch):
        sigma_space = [0, math.sqrt(X.shape[1])]

        params.mutation = trial.suggest_categorical('mutation', ['HalfnormIncrease', 'Normal'])
        params.mutation = getattr(es.mutation, params.mutation)()
        params.mutation__sigma = trial.suggest_float('mutation__sigma', *sigma_space)
        #
        if isinstance(params.mutation, es.mutation.HalfnormIncrease):
            params.init = rule.initialization.MeanInit()
            params.delay = 40
        else:
            params.init = rule.initialization.HalfnormInit()
            params.init__sigma = trial.suggest_float('init__sigma', *sigma_space)
            params.delay = 1

    params = global_params | dataset_params.get(problem, {})

    experiment = Experiment(name=f'Rule Generation Tuning', params=params, verbose=10)

    tuner = OptunaTuner(X_train=X_train, y_train=y_train, **shared_tuning_params)
    experiment.with_tuning(rule_generation_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(2)
    experiment.with_random_states(random_states, n_jobs=2)

    evaluation = CrossValidateTest(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   random_state=random_state, verbose=10)
    experiment.perform(evaluation=evaluation, cv=8, n_jobs=8)

    mlflow.set_experiment("RG Tuning")
    log_experiment(experiment)


if __name__ == '__main__':
    run()
