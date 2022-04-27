import math

import click
import mlflow
from optuna import Trial
from sklearn.utils import Bunch, shuffle
from suprb import rule
from suprb.optimizer.rule import mutation
from suprb.optimizer.solution import ga
from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from runs.solution_composition.shared_config import shared_tuning_params, load_dataset, global_params, dataset_params, \
    random_state

@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
def run(problem: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    # TODO Set all params to be adjusted

    @param_space()
    def rule_generation_space(trial: Trial, params: Bunch):
        """
        sigma_space = [0, math.sqrt(X.shape[1])]
        params.rule_generation__n_iter = trial.suggest_int('n_iter_es', 4, 16)
        params.solution_composition__n_iter = trial.suggest_int('n_iter_ga', 4, 16)
        params.n_rules = trial.suggest_int('trial', 2, 3)
        params.solution_composition__selection = trial.suggest_categorical('selection',
                                                     ['RouletteWheel', 'Tournament', 'LinearRank', 'Random'])
        params.solution_composition__selection = getattr(ga.selection, params.solution_composition__selection)()
        if isinstance(params.solution_composition__selection, ga.selection.Tournament):
            params.solution_composition__selection__k = trial.suggest_int('selection__k', 3, 10)
        """
        params.rule_generation__delay = trial.suggest_int('delay', 1, 100)


    params = global_params | dataset_params.get(problem, {})

    experiment = Experiment(name=f'{problem} RG Tuning', params=params, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **shared_tuning_params,
                        scoring='fitness')
    experiment.with_tuning(rule_generation_space, tuner=tuner)

    experiment.perform(evaluation=None)

    mlflow.set_experiment("Representation Tuning")
    log_experiment(experiment)


if __name__ == '__main__':
    run()