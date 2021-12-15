from typing import Union, Callable, Any

import numpy as np
from sklearn.base import BaseEstimator
from skopt import forest_minimize, gbrt_minimize, gp_minimize

from .base import ParameterTuner


class SkoptTuner(ParameterTuner):
    """
    Parameter tuning using either

    - Gaussian Processes (gp),

    - Gradient-boosted Trees (gbrt),

    - Random Forest (forest).
    """

    def __init__(self,
                 estimator: BaseEstimator,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 scoring: Union[str, Callable],
                 parameter_space: dict[str, Any],
                 tuner: str = 'gp',
                 **kwargs):
        super().__init__(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            parameter_space=parameter_space,
            **kwargs
        )

        self.tuner = tuner

    @staticmethod
    def _get_optimizer(tuner: str) -> Callable:
        return {
            'forest': forest_minimize,
            'gbrt': gbrt_minimize,
            'gp': gp_minimize
        }[tuner]

    def _optimize(self, objective: Callable[[dict], float]) -> list:
        return (self._get_optimizer(self.tuner))(
            func=objective,
            dimensions=self.parameter_space.values(),
            n_calls=self.n_calls,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            callback=self.callback,
            verbose=self.verbose,
        )
