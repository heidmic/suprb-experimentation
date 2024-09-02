import numpy as np
import mlflow

from problems import scale_X_y
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm


from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger

import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split

from suprb import SupRB
from suprb.utils import check_random_state
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm
import suprb.json as suprb_json

random_state = 42

def load_higdon_gramacy_lee(n_samples=1000, noise=0, random_state=None):
    random_state_ = check_random_state(random_state)

    X = np.linspace(0, 20, num=n_samples)
    y = np.zeros(n_samples)

    y[X < 10] = np.sin(np.pi * X[X < 10] / 5) + 0.2 * np.cos(4 * np.pi * X[X < 10] / 5)
    y[X >= 10] = X[X >= 10] / 10 - 1

    y += random_state_.normal(scale=noise, size=n_samples)
    X = X.reshape((-1, 1))

    return sklearn.utils.shuffle(X, y, random_state=random_state)


def run():
    product_twins = np.load('arrays.npz')
    # X, y = product_twins['arr1'], product_twins['arr2']

    X, y = load_higdon_gramacy_lee(noise=0.1, random_state=random_state)

    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator =  SupRB(rule_generation=ES1xLambda(),
                  solution_composition=GeneticAlgorithm(),
                  logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]))

    experiment_name = f'Adeles Run'

    print(experiment_name)
    experiment = Experiment(name=experiment_name,  verbose=10)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    evaluation = CrossValidate(
        estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(
        n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
