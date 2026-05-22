import click
import mlflow
from optuna import Trial
from sklearn.utils import Bunch, shuffle
from suprb.optimizer.solution import ga
from suprbopt.solution import gwo, aco, pso, abc

from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import solution_composition_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from shared_config import load_dataset, shared_tuning_params, global_params, get_optimizer, dataset_params, random_state


def ga_space(trial: Trial, params: Bunch):
    params.selection = trial.suggest_categorical("selection", ["RouletteWheel", "Tournament", "LinearRank", "Random"])
    params.selection = getattr(ga.selection, params.selection)()

    if isinstance(params.selection, ga.selection.Tournament):
        params.selection__k = trial.suggest_int("selection__k", 3, 10)

    params.crossover = trial.suggest_categorical("crossover", ["NPoint", "Uniform"])
    params.crossover = getattr(ga.crossover, params.crossover)()

    if isinstance(params.crossover, ga.crossover.NPoint):
        params.crossover__n = trial.suggest_int("crossover__n", 1, 10)

    params.mutation__mutation_rate = trial.suggest_float("mutation_rate", 0, 0.1)
    params.elitist_ratio = trial.suggest_float("elitist_ratio", 0, 0.2)


def gwo_space(trial: Trial, params: Bunch):
    params.position = trial.suggest_categorical("position", ["Sigmoid", "Crossover"])
    params.position = getattr(gwo.position, params.position)()
    params.n_leaders = trial.suggest_int("n_leaders", 1, global_params.solution_composition__population_size // 2)


def aco_space(trial: Trial, params: Bunch):
    params.builder = trial.suggest_categorical("builder", ["Binary", "Complete"])
    params.builder = getattr(aco.builder, params.builder)()
    params.builder.alpha = trial.suggest_float("alpha", 0.5, 5)
    params.builder.beta = trial.suggest_float("beta", 0.5, 5)

    params.evaporation_rate = trial.suggest_float("evaporation_rate", 0, 0.9)
    params.selection__n = trial.suggest_int("selection__n", 1, global_params.solution_composition__population_size // 2)


def pso_space(trial: Trial, params: Bunch):
    params.movement = trial.suggest_categorical("movement", ["Sigmoid", "SigmoidQuantum", "BinaryQuantum"])
    params.movement = getattr(pso.movement, params.movement)()

    params.a_min = trial.suggest_float("a_min", 0, 3)
    params.a_max = trial.suggest_float("a_max", params.a_min, 3)

    if isinstance(params.movement, pso.movement.Sigmoid):
        params.movement.b = trial.suggest_float("b", 0, 3)
        params.movement.c = trial.suggest_float("c", 0, 3)
    elif isinstance(params.movement, pso.movement.BinaryQuantum):
        params.movement.p_learning = trial.suggest_float("p_learning", 0.01, 1)
        params.movement.n_attractors = trial.suggest_int(
            "n_attractors", 1, global_params.solution_composition__population_size // 2
        )


def abc_space(trial: Trial, params: Bunch):
    params.food = trial.suggest_categorical("food", ["Sigmoid", "Bitwise", "DimensionFlips"])
    params.food = getattr(abc.food, params.food)()

    params.trials_limit = trial.suggest_int("trials_limit", 1, global_params.solution_composition__n_iter)

    if isinstance(params.food, abc.food.DimensionFlips):
        params.food.flip_rate = trial.suggest_float("flip_rate", 0.01, 1)


@click.command()
@click.option("-p", "--problem", type=click.STRING, default="concrete_strength")
@click.option("-o", "--optimizer", type=click.STRING, default="ga")
def run(problem: str, optimizer: str):
    print(f"Problem is {problem}, optimizer is {optimizer}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    params = global_params | dataset_params.get(problem, {}) | {"solution_composition": get_optimizer(optimizer)}

    experiment = Experiment(name=f"{optimizer.upper()} Tuning", params=params, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, scoring="fitness", **shared_tuning_params)
    experiment.with_tuning(solution_composition_space(globals()[f"{optimizer}_space"]), tuner=tuner)

    experiment.perform(evaluation=None)

    mlflow.set_experiment(f"{problem} Tuning")
    log_experiment(experiment)


if __name__ == "__main__":
    run()
