import pandas as pd
import numpy as np
import sys
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.utils import shuffle

from experiments import Experiment
from experiments.mlflow import log_experiment
from problems import scale_X_y
from problems.datasets import load_concrete_strength
from suprb.rule.matching import MatchingFunction, OrderedBound, UnorderedBound, CenterSpread, MinPercentage, \
    GaussianKernelFunction
import suprb.optimizer.rule.mutation
from suprb import SupRB
from suprb import rule
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es

if __name__ == '__main__':
    random_state = 42

    X, y = load_concrete_strength()
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    models = [
        SupRB(
            rule_generation=es.ES1xLambda(
                operator='&',
                init=rule.initialization.NormalInit(sigma=np.array([0.00000001,1]), fitness=rule.fitness.VolumeWu(alpha=0.8)),
                mutation=suprb.optimizer.rule.mutation.Uniform(sigma=np.array([1,1]))
            ),
            solution_composition=ga.GeneticAlgorithm(
                n_iter=64,
                crossover=ga.crossover.Uniform(),
                selection=ga.selection.Tournament(),
                mutation=ga.mutation.BitFlips(),
            ),
            matching_type=GaussianKernelFunction(np.array([]), np.array([])),
            #matching_type=OrderedBound(np.array([])),
            n_iter=16,
            n_rules=4,
            logger=StdoutLogger(),
            random_state=random_state,
        )
    ]
    models = {model.__class__.__name__: model for model in models}


    experiment = Experiment(name=f'HEC ES Tuning & Experimentation', verbose=10)
    log_experiment(experiment)


