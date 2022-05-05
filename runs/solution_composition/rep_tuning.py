import math
import os
import click
import mlflow
from optuna import Trial
from sklearn.utils import Bunch, shuffle
from suprb.optimizer.rule import mutation
from suprb.optimizer.solution import ga
from suprb import rule
from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from runs.solution_composition.shared_config import shared_tuning_params, load_dataset, \
    global_params, individual_dataset_params, random_state


@param_space()
def rule_generation_space(trial: Trial, params: Bunch):
    # Evolution Strategy - Mutation, Mutation_sigma, Initialization, Init_sigma, Delay (delta) and fitness_alpha
    sigma_space = [0, 3]

    params.rule_generation__mutation = \
        trial.suggest_categorical('mutation', ['Normal', 'HalfnormIncrease', 'UniformIncrease'])
    params.rule_generation__mutation = getattr(mutation, params.rule_generation__mutation)()
    params.rule_generation__mutation__sigma = trial.suggest_float('sigma_mutate', *sigma_space)

    params.rule_generation__init = \
        trial.suggest_categorical('initialization', ['MeanInit', 'NormalInit'])
    params.rule_generation__init = getattr(rule.initialization, params.rule_generation__init)()
    if isinstance(params.rule_generation__init, rule.initialization.NormalInit):
        params.rule_generation__init__sigma = trial.suggest_float('sigma_init', *sigma_space)

    alpha_space = [0, 0.1]
    params.rule_generation__init__fitness__alpha = trial.suggest_float('alpha', *alpha_space)

    # Delay being larger than n_iter of rule_generation is pointless?
    params.rule_generation__delay = trial.suggest_int('delay', low=1, high=100)

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


datasets = {0: 'superconductivity', 1: 'protein_structure', 2: 'online_news',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant', 5: 'airfoil_self_noise'}


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='combined_cycle_power_plant')
def run(problem: str):
    # my_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # problem = datasets.get(my_index)
    print(f"Problem is {problem}")
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    params = global_params | individual_dataset_params.get(problem, {})

    experiment = Experiment(name=f'{problem} General Tuning', params=params, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **shared_tuning_params,
                        scoring='fitness')
    experiment.with_tuning(rule_generation_space, tuner=tuner)

    experiment.perform(evaluation=None)

    mlflow.set_experiment(problem)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
