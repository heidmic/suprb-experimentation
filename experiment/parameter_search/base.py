from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_validate
from skopt.utils import use_named_args, point_asdict


class ParameterTuner(metaclass=ABCMeta):
    """A generic parameter tuning method."""

    tuning_result_: Any
    tuned_params_: dict

    def __init__(self, estimator: BaseEstimator = None,
                 X_train: np.ndarray = None,
                 y_train: np.ndarray = None,
                 scoring: Union[str, Callable] = 'r2',
                 parameter_space: dict[str, Any] = None,
                 n_calls: int = 32,
                 callback: Union[Callable, list[Callable]] = None,
                 cv: int = None,
                 n_jobs_cv: int = None,
                 n_jobs: int = None,
                 verbose: int = 0,
                 random_state: int = None
                 ):
        self.estimator = estimator
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.parameter_space = parameter_space
        self.n_calls = n_calls
        self.callback = callback
        self.cv = cv
        self.n_jobs_cv = n_jobs_cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _init_parameter_space(self):
        for name, dimension in self.parameter_space.items():
            dimension.name = name

    @staticmethod
    def _validate_sklearn_metric(metric: str) -> bool:
        return metric in sklearn.metrics.SCORERS.keys()

    @staticmethod
    def _validate_own_metric(self, metric: str) -> bool:
        pass

    @staticmethod
    def _compute_metric(metric: Union[Callable, str], scores: dict) -> list:
        if callable(metric):
            return metric(scores['estimator'])
        else:
            pass

    def tune(self) -> dict:
        self._init_parameter_space()

        estimator = clone(self.estimator)
        estimator.set_params(random_state=self.random_state)

        @use_named_args(self.parameter_space.values())
        def objective(**params):
            estimator.set_params(**params)
            scores = cross_validate(
                estimator,
                self.X_train,
                self.y_train,
                cv=self.cv,
                scoring=self.scoring if self._validate_sklearn_metric(self.scoring) else None,
                n_jobs=self.n_jobs_cv,
                return_estimator=True,
                verbose=self.verbose,
            )

            if self._validate_sklearn_metric(self.scoring):
                score = scores['test_score']
            elif self.scoring == 'fitness':
                score = [estimator.elitist_.fitness_ for estimator in scores['estimator']]

            return -np.mean(score)

        self.tuning_result_ = self._optimize(objective=objective)
        self.tuned_params_ = point_asdict(self.parameter_space, self.tuning_result_.x)
        return self.tuned_params_

    @abstractmethod
    def _optimize(self, objective: Callable[[dict], float]) -> list:
        pass
