import click
import mlflow
import numpy as np
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from problems import scale_X_y
from shared_config import (
    load_dataset,
    global_params,
    estimator,
    random_state,
    dataset_params,
    get_optimizer,
    optimizer_params,
)


@click.command()
@click.option("-p", "--problem", type=click.STRING, default="concrete_strength")
@click.option("-o", "--optimizer", type=click.STRING, default="ga")
def run(problem: str, optimizer: str):
    print(f"Problem is {problem}, optimizer is {optimizer}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)

    params = (
        global_params
        | dataset_params.get(problem, {})
        | {"solution_composition": get_optimizer(optimizer)}
        | optimizer_params.get(problem, {}).get(optimizer, {})
    )

    experiment = Experiment(name=f"{optimizer.upper()} Evaluation", params=params, verbose=10)

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    # Evaluation
    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(problem)
    log_experiment(experiment)


if __name__ == "__main__":
    run()
