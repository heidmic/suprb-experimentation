from typing import Union, Callable, Any, Optional
import os
import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from .base import ParameterTuner


class OptunaTuner(ParameterTuner):
    """
    Parameter tuning using Optuna.

    Supports a shared PostgreSQL study: pass `storage` and `study_name`
    explicitly (e.g. from the worker script via environment variables) so
    multiple workers can contribute trials to the same study concurrently.
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scoring: Union[str, Callable] = "r2",
        callback: Union[Callable, list[Callable]] = None,
        tuner: str = "tpe",
        timeout: Optional[float] = None,
        study_name: str = "NoName",
        storage: Optional[str] = None,  # ← NEW: explicit storage URL
        **kwargs,
    ):
        super().__init__(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            **kwargs,
        )
        self.callback = callback
        self.tuner = tuner
        self.timeout = timeout
        self.study_name = study_name
        self.storage = storage  # if None, _resolve_storage() is used as fallback

    def get_params(self):
        return super().get_params() | self._get_params(["timeout"])

    @staticmethod
    def _get_optimizer(tuner: str) -> Callable:
        return {
            "tpe": optuna.samplers.TPESampler,
            "cma-es": optuna.samplers.CmaEsSampler,
        }[tuner]

    @staticmethod
    def _resolve_storage() -> str:
        """
        Fallback storage resolution when no explicit URL was passed.

        Priority:
          1. PG_SOCKET_DIR    — set by coordinator for local/interactive use
          2. SCRATCH          — legacy: old per-task postgres in scratch
          3. DEVENV_STATE     — legacy: local devenv shell
          4. localhost        — last resort
        """
        url = os.environ.get("OPTUNA_STORAGE")
        if url:
            return url

        for env_var, sub in (("PG_SOCKET_DIR", ""), ("SCRATCH", "/postgres"), ("DEVENV_STATE", "/postgres")):
            val = os.environ.get(env_var)
            if val:
                return f"postgresql+psycopg2:///optuna_db?host={val}{sub}"

        return "postgresql+psycopg2://localhost/optuna_db"

    def __call__(self, parameter_space: Callable, local_params: dict) -> tuple[dict, Any]:
        old_objective = self.generate_objective_function(**local_params)

        def objective(trial: optuna.Trial):
            params = parameter_space(trial)
            return old_objective(**params)

        storage_url = self.storage or self._resolve_storage()

        sampler = self._get_optimizer(self.tuner)(seed=self.random_state)

        study = optuna.create_study(
            sampler=sampler,
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,  # workers safely join an existing study
        )

        study.optimize(
            func=objective,
            n_trials=self.n_calls,
            n_jobs=self.n_jobs if self.n_jobs is not None else 1,
            timeout=self.timeout,
            callbacks=self.callback if isinstance(self.callback, list) else ([self.callback] if self.callback else None),
        )

        self.tuned_params_ = parameter_space(study.best_trial)
        self.tuning_result_ = Bunch()
        self.tuning_result_.objective_history = [trial.value for trial in study.trials]
        self.tuning_result_.params_history = [trial.params for trial in study.trials]

        return self.tuned_params_, self.tuning_result_
