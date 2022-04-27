import click
import mlflow
import os
from optuna import Trial
from sklearn.utils import Bunch, shuffle
from suprb.optimizer.solution import ga

from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import solution_composition_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from shared_config import load_dataset, shared_tuning_params, global_params, get_optimizer, dataset_params, random_state


def ga_space(trial: Trial, params: Bunch):
    params.selection = trial.suggest_categorical('selection',
                                                 ['RouletteWheel', 'Tournament', 'LinearRank', 'Random'])
    params.selection = getattr(ga.selection, params.selection)()

    if isinstance(params.selection, ga.selection.Tournament):
        params.selection__k = trial.suggest_int('selection__k', 3, 10)

    params.crossover = trial.suggest_categorical('crossover', ['NPoint', 'Uniform'])
    params.crossover = getattr(ga.crossover, params.crossover)()

    if isinstance(params.crossover, ga.crossover.NPoint):
        params.crossover__n = trial.suggest_int('crossover__n', 1, 10)

    params.mutation__mutation_rate = trial.suggest_float('mutation_rate', 0, 0.1)
    params.elitist_ratio = trial.suggest_float('elitist_ratio', 0, 0.2)


# Used to control dataset which dataset gets selected through slurm
datasets = {0: 'airfoil_self_noise', 1: 'combined_cycle_power_plant', 2: 'concrete_strength',
            3: 'online_news', 4: 'protein_structure', 5: 'superconductivity'}


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='concrete_strength')
@click.option('-o', '--optimizer', type=click.STRING, default='ga')
def run(problem: str, optimizer: str):
    # my_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # problem = datasets.get(my_index)
    print(f"Problem is {problem}, optimizer is {optimizer}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    params = global_params | dataset_params.get(problem, {}) | {'solution_composition': get_optimizer(optimizer)}

    experiment = Experiment(name=f'{optimizer.upper()} Tuning', params=params, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, scoring='fitness', **shared_tuning_params)
    experiment.with_tuning(solution_composition_space(globals()[f"{optimizer}_space"]), tuner=tuner)

    experiment.perform(evaluation=None)

    mlflow.set_experiment(f"{problem} Tuning")
    log_experiment(experiment)


if __name__ == '__main__':
    run()
