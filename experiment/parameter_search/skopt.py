from typing import Union, Callable, Any

import numpy as np
from sklearn.base import BaseEstimator
from skopt import forest_minimize, gbrt_minimize, gp_minimize
from skopt.utils import use_named_args, point_asdict

from .base import ParameterTuner


class SkoptTuner(ParameterTuner):
    """
    Parameter tuning using either
    - Gaussian Processes (gp),
    - Gradient-boosted Trees (gbrt),
    - Random Forest (forest).
    """

    def __init__(
            self,
            estimator: BaseEstimator,
            X_train: np.ndarray,
            y_train: np.ndarray,
            scoring: Union[str, Callable] = 'r2',
            tuner: str = 'gp',
            **kwargs
    ):
        super().__init__(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
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

    def __call__(self, parameter_space: dict[str, Any]) -> tuple[dict, Any]:
        # Sets the key of the dict as name, because `skopt` handles this weirdly
        for name, dimension in parameter_space.items():
            dimension.name = name

        # Convert the objective function to accept a list rather than a dict
        objective = use_named_args(parameter_space.values())(self.generate_objective_function())

        # Perform the tuning
        self.tuning_result_ = (self._get_optimizer(self.tuner))(
            func=objective,
            dimensions=parameter_space.values(),
            n_calls=self.n_calls,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            callback=self.callback,
            verbose=self.verbose,
        )

        # Convert the list back to a dict, such that it can be used with `set_params`
        self.tuned_params_ = point_asdict(parameter_space, self.tuning_result_.x)
        return self.tuned_params_, self.tuning_result_
