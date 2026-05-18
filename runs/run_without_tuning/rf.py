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

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm


from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

random_state = 42


def run():
    X = pd.read_parquet("new_data/features_preselection.parq")
    y = pd.read_parquet("new_data/target.parq").iloc[:, 0]

    X = X.values
    y = y.values.flatten()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    estimator = RandomForestRegressor(random_state=random_state)

    experiment_name = f"Random Forest"
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
