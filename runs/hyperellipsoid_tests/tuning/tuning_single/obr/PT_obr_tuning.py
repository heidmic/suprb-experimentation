import click
import mlflow
import numpy as np
from optuna import Trial
from sklearn.utils import Bunch, shuffle
from suprb.optimizer.rule import mutation
from suprb.optimizer.solution import ga
from suprb import rule
from suprb.rule.matching import OrderedBound

from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from runs.hyperellipsoid_tests.configurations.shared_config import shared_tuning_params, load_dataset, \
    global_params, individual_dataset_params, random_state


@param_space()
def che_space(trial: Trial, params: Bunch):
    # Matching type
    params.matching_type = OrderedBound(np.array([]))

    # Evolution Strategy - Mutation, Mutation_sigma, Initialization, Init_sigma, Delay (delta) and fitness_alpha
    sigma_space = [0, 3]

    params.rule_generation__mutation = \
        trial.suggest_categorical('mutation', ['Normal', 'HalfnormIncrease', 'Uniform', 'UniformIncrease'])
    params.rule_generation__mutation = getattr(mutation, params.rule_generation__mutation)()


    params.rule_generation__init = \
        trial.suggest_categorical('initialization', ['MeanInit', 'NormalInit'])
    params.rule_generation__init = getattr(rule.initialization, params.rule_generation__init)()

    alpha_space = [0, 0.1]
    params.rule_generation__init__fitness__alpha = trial.suggest_float('alpha', *alpha_space)

    params.rule_generation__operator = \
        trial.suggest_categorical('operator', ['&', '+'])

    if params.rule_generation__operator in ('+', ','):
        params.rule_generation__n_iter = trial.suggest_int('n_iter_es', low=1, high=50)
    else:
        params.rule_generation__delay = trial.suggest_int('delay', low=1, high=25)

    # Genetic Algorithm - Selection, TournamentSelection - k, Crossover, Crossover_n, mutation_rate
    params.solution_composition__selection = \
        trial.suggest_categorical('selection', ['RouletteWheel', 'Tournament', 'LinearRank', 'Random'])
    params.solution_composition__selection = getattr(ga.selection, params.solution_composition__selection)()
    if isinstance(params.solution_composition__selection, ga.selection.Tournament):
        params.solution_composition__selection__k = trial.suggest_int('selection__k', 3, 10)

    params.solution_composition__crossover = trial.suggest_categorical('crossover', ['NPoint', 'Uniform'])
    params.solution_composition__crossover = getattr(ga.crossover, params.solution_composition__crossover)()
    if isinstance(params.solution_composition__crossover, ga.crossover.NPoint):
        params.solution_composition__crossover__n = trial.suggest_int('crossover__n', 1, 10)

    params.solution_composition__mutation__mutation_rate = trial.suggest_float('mutation_rate', 0, 0.1)


datasets = {0: 'parkinson_total'}


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='parkinson_total')
def run(problem: str):
    print(f"Problem is {problem}")
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    params = global_params | individual_dataset_params.get(problem, {})

    experiment = Experiment(name=f'{problem} General Tuning', params=params, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **shared_tuning_params, scoring='fitness')
    experiment.with_tuning(che_space, tuner=tuner)

    experiment.perform(evaluation=None)
    mlflow.set_experiment(problem)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
