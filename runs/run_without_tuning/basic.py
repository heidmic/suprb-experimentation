import numpy as np
import mlflow

from problems import scale_X_y
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment

from suprb import SupRB, rule
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm


from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import suprb
from sklearn.linear_model import Ridge
from suprb.optimizer.rule import es, origin, mutation
import click
from suprb.wrapper import SupRBWrapper
import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es, origin, mutation
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd

random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets

    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option("-n", "--experiment_name", type=click.STRING, default="SupRB")
@click.option("-w", "--fitness_weight", type=click.FLOAT, default=0.3)
@click.option("-s", "--scaler_type", type=click.BOOL, default=True)
@click.option("-i", "--n_iter", type=click.INT, default=32)
@click.option("-p", "--problem", type=click.STRING, default="airfoil_self_noise")
def run(experiment_name: str, fitness_weight: float, scaler_type: bool, n_iter: int, problem: str):
    X = pd.read_parquet("new_data/features_preselection.parq")
    y = pd.read_parquet("new_data/target.parq").iloc[:, 0]

    X = X.values
    y = y.values.flatten()

    if scaler_type:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)
    else:
        X, y = scale_X_y(X, y)
        X, y = shuffle(X, y, random_state=random_state)

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        n_iter=n_iter,
        n_rules=16,
        n_jobs=1,
        rule_discovery=ES1xLambda(
            n_jobs=1,
            origin_generation=origin.SquaredError(),
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.VolumeWu(), model=Ridge(alpha=0.01, random_state=random_state)
            ),
        ),
        solution_composition=GeneticAlgorithm(
            n_iter=64,
            n_jobs=1,
            init=suprb.solution.initialization.RandomInit(
                fitness=suprb.solution.fitness.ComplexityWu(alpha=fitness_weight)
            ),
        ),
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
    )

    estimator = SupRBWrapper(
        rule_discovery__mutation__sigma=2.53261854608031,
        rule_discovery__delay=134,
        rule_discovery__init__fitness__alpha=0.043582602456505595,
    )

    jobs = 8

    print(experiment_name)
    experiment = Experiment(name=experiment_name, verbose=10)

    random_states = np.random.SeedSequence(random_state).generate_state(jobs)
    experiment.with_random_states(random_states, n_jobs=jobs)

    evaluation = CrossValidate(
        estimator=estimator,
        X=X,
        y=y,
        random_state=random_state,
        verbose=10,
    )

    experiment.perform(
        evaluation, cv=ShuffleSplit(n_splits=jobs, test_size=0.25, random_state=random_state), n_jobs=jobs
    )

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == "__main__":
    run()
