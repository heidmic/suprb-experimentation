from typing import Union, Callable, Any

import numpy as np
import optuna
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
from sqlalchemy import create_engine, inspect
from optuna.storages._rdb.models import BaseModel


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
            scoring: Union[str, Callable] = 'r2',
            callback: Union[Callable, list[Callable]] = None,
            tuner: str = 'tpe',
            timeout: float = None,
            study_name: str = "NoName",
            **kwargs
    ):
        super().__init__(
            estimator=estimator,
            X_train=X_train,
            y_train=y_train,
            scoring=scoring,
            **kwargs
        )

        self.callback = callback
        self.tuner = tuner
        self.timeout = timeout
        self.study_name = study_name

    def get_params(self):
        return super().get_params() | self._get_params(['timeout'])

    @staticmethod
    def _get_optimizer(tuner: str) -> Callable:
        return {
            'tpe': optuna.samplers.TPESampler,
            'cma-es': optuna.samplers.CmaEsSampler,
        }[tuner]

    def __call__(self, parameter_space: Callable, local_params: dict) -> tuple[dict, Any]:
        old_objective = self.generate_objective_function(**local_params)

        def objective(trial: optuna.Trial):
            params = parameter_space(trial)
            return old_objective(**params)

        sampler = self._get_optimizer(self.tuner)(seed=self.random_state)

        def create_optuna_study(study_name, sampler, db_file='suprb_optuna.db'):
            storage_name = f'sqlite:///{db_file}'
            engine = create_engine(storage_name)

            inspector = inspect(engine)
            # if 'studies' not in inspector.get_table_names() and 'alembic_version' not in inspector.get_table_names():
            #     storage = optuna.storages.RDBStorage(url=storage_name)
            # else:
            #     storage = optuna.storages.RDBStorage(url=storage_name)

            # return optuna.create_study(sampler=sampler, study_name=study_name, storage=storage)
            if 'studies' not in inspector.get_table_names():
                # Initialize the Optuna storage if tables do not exist
                storage = optuna.storages.RDBStorage(url=storage_name)
                # Ensure the BaseModel metadata is created
                BaseModel.metadata.create_all(engine)
            else:
                # Use the existing Optuna storage
                storage = optuna.storages.RDBStorage(url=storage_name)

            # Create or load the study
            study = optuna.create_study(sampler=sampler, study_name=study_name, storage=storage)

            return study

        study = create_optuna_study(self.study_name, sampler)

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
