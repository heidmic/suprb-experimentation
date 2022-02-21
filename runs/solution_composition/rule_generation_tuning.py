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
from runs.solution_composition.shared_config import shared_tuning_params, load_dataset, global_params, dataset_params, \
    random_state
import suprb2


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='concrete_strength')
def run(problem: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    @param_space('rule_generation')
    def rule_generation_space(trial: Trial, params: Bunch):
        sigma_space = [0, 1]

        # params.mutation = trial.suggest_categorical('mutation', ['Uniform'])
        # params.mutation = getattr(suprb2.optimizer.rule.mutation, params.mutation)()
        # params.mutation__sigma = trial.suggest_float('mutation__sigma', *sigma_space)
        # params.init__fitness__alpha = trial.suggest_float('init__fitness__alpha', 0.05, 1)

        params.n_iter = trial.suggest_int('n_iter', 1, 500)

        # if isinstance(params.mutation, es.mutation.HalfnormIncrease):
        #     params.init = rule.initialization.MeanInit()
        # else:
        #     params.init = rule.initialization.HalfnormInit()
        #     params.init__sigma = trial.suggest_float('init__sigma', *sigma_space)
        pass

    params = global_params | dataset_params.get(problem, {})

    experiment = Experiment(name=f'{problem} RG Tuning', params=params, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **shared_tuning_params)
    experiment.with_tuning(rule_generation_space, tuner=tuner)

    experiment.perform(evaluation=None)

    mlflow.set_experiment("RG Tuning")
    log_experiment(experiment)


if __name__ == '__main__':
    run()
