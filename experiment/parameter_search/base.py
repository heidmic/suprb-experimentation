from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Any

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_validate

from . import metrics


def _validate_sklearn_metric(metric: str) -> bool:
    return metric in sklearn.metrics.SCORERS.keys()


def _validate_own_metric(metric: str) -> bool:
    return metric in metrics.__all__


class ParameterTuner(metaclass=ABCMeta):
    """A generic parameter tuning method."""

    tuning_result_: Any
    tuned_params_: dict

    parameter_space: dict

    def __init__(self,
                 estimator: BaseEstimator,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 scoring: Union[str, Callable],
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
        self.n_calls = n_calls
        self.callback = callback
        self.cv = cv
        self.n_jobs_cv = n_jobs_cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def generate_objective_function(self):
        """The default objective function performs cross-validation and uses the average of the scoring as value."""

        estimator = clone(self.estimator)
        estimator.set_params(random_state=self.random_state)

        def objective(**params):
            estimator.set_params(**params)
            scores = cross_validate(
                estimator,
                self.X_train,
                self.y_train,
                cv=self.cv,
                scoring=self.scoring if _validate_sklearn_metric(self.scoring) else None,
                n_jobs=self.n_jobs_cv,
                return_estimator=True,
                verbose=self.verbose,
                error_score='raise',
            )

            if _validate_sklearn_metric(self.scoring):
                score = scores['test_score']
            elif _validate_own_metric(self.scoring):
                score = [getattr(metrics, self.scoring)(_estimator) for _estimator in scores['estimator']]
            else:
                raise ValueError('invalid scoring metric')

            return -np.mean(score)

        return objective

    @abstractmethod
    def __call__(self, parameter_space: dict[str, Any]) -> tuple[dict, Any]:
        pass
