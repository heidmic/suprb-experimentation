from typing import Union, Callable, Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from datetime import datetime

from .base import ParameterTuner


class OptunaTuner(ParameterTuner):
    """
    Parameter tuning using optuna.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: Union[str, Callable] = "r2",
        callback: Union[Callable, list[Callable]] = None,
        tuner: str = "tpe",
        timeout: float = None,
        study_name: str = "NoName",
        **kwargs,
    ):
        super().__init__(estimator=estimator, X_train=X_train, y_train=y_train, scoring=scoring, **kwargs)

        self.callback = callback
        self.tuner = tuner
        self.timeout = timeout
        self.study_name = study_name

    def get_params(self):
        return super().get_params() | self._get_params(["timeout"])

    @staticmethod
    def _get_optimizer(tuner: str) -> Callable:
        return {
            "tpe": optuna.samplers.TPESampler,
            "cma-es": optuna.samplers.CmaEsSampler,
        }[tuner]

    def __call__(self, parameter_space: Callable, local_params: dict) -> tuple[dict, Any]:
        old_objective = self.generate_objective_function(**local_params)

        def objective(trial: optuna.Trial):
            params = parameter_space(trial)
            return old_objective(**params)

        sampler = self._get_optimizer(self.tuner)(seed=self.random_state)

        storage_name = f'sqlite:///suprb_optuna_{datetime.now().strftime("%Y-%m-%d")}.db'
        study = optuna.create_study(
            sampler=sampler,
            study_name=self.study_name,
            # storage=storage_name,
            load_if_exists=True,
        )

        study.optimize(
            func=objective,
            n_trials=self.n_calls,
            n_jobs=self.n_jobs if self.n_jobs is not None else 1,
            timeout=self.timeout,
        )

        self.tuned_params_ = parameter_space(study.best_trial)

        self.tuning_result_ = Bunch()
        self.tuning_result_.objective_history = [trial.value for trial in study.trials]
        self.tuning_result_.params_history = [trial.params for trial in study.trials]

        return self.tuned_params_, self.tuning_result_
