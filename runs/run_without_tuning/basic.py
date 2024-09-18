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

random_state = 42


def run():
    X = pd.read_parquet('new_data/features_preselection.parq')
    y = pd.read_parquet('new_data/target.parq').iloc[:, 0]

    X = X.values
    y = y.values.flatten()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    estimator = SupRB(n_iter=64,
                      n_rules=16,
                      n_jobs=20,
                      rule_generation=ES1xLambda(n_jobs=20,
                                                 origin_generation=origin.SquaredError(),
                                                 init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                                                                   model=Ridge(alpha=0.01, random_state=random_state))),
                      solution_composition=GeneticAlgorithm(n_iter=64,
                                                            n_jobs=20,
                                                            init=suprb.solution.initialization.RandomInit(fitness=suprb.solution.fitness.ComplexityWu(alpha=0.1))),
                      logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]))

    experiment_name = f'SupRB'
    jobs = 8

    print(experiment_name)
    experiment = Experiment(name=experiment_name, verbose=10)

    random_states = np.random.SeedSequence(random_state).generate_state(jobs)
    experiment.with_random_states(random_states, n_jobs=jobs)

    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10,)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=jobs, test_size=0.25, random_state=random_state), n_jobs=jobs)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
