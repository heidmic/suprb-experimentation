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


random_state = 42

def run():
    product_twins = np.load('arrays.npz')
    X, y = product_twins['arr1'], product_twins['arr2']

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
