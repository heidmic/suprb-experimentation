import click
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from suprb2.optimizer.individual import ga
from suprb2opt.individual import gwo, aco, pso, abc, rs

from experiments import Experiment
from experiments.evaluation import CrossValidate, CrossValidateTest
from experiments.mlflow import log_experiment
from problems import scale_X_y
from shared_config import load_dataset, global_params, estimator, random_state, dataset_params


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-o', '--optimizer', type=click.STRING, default='ga')
def run(problem: str, optimizer: str):
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    ga_optimizer = ga.GeneticAlgorithm(
        # TODO
    )

    gwo_optimizer = gwo.GreyWolfOptimizer(
        # TODO
    )

    aco_optimizer = aco.AntColonyOptimization(
        # TODO
    )

    pso_optimizer = pso.ParticleSwarmOptimization(
        # TODO
    )

    abc_optimizer = abc.ArtificialBeeColonyAlgorithm(
        # TODO
    )

    rs_optimizer = rs.RandomSearch()

    params = global_params | dataset_params.get(problem, {}) | {
        'individual_optimizer': locals()[f"{optimizer}_optimizer"]}

    experiment = Experiment(name=f'{optimizer.upper()} Evaluation', params=params, verbose=10)

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    # Evaluation
    evaluation = CrossValidateTest(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=8, n_jobs=8)

    mlflow.set_experiment(problem)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
