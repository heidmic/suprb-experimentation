from __future__ import annotations

import copy
from abc import ABCMeta, abstractmethod
from typing import Union, Iterator, Generator

import numpy as np
from sklearn.base import BaseEstimator, clone

from experiment.parameter_search import ParameterTuner


class Experiment(metaclass=ABCMeta):
    """Base class for experiments."""

    def set_params(self, **params):
        """Set the parameters of this and nested experiments."""
        for key, value in params.items():
            if hasattr(self, key) and getattr(self, key) is None:
                setattr(self, key, copy.copy(value))

    def clone(self) -> Experiment:
        """
        Return a identical clone of the experiment, including nested experiments.
        Note that we don't want to use `copy.deepcopy()`, because numpy arrays etc should not be copied.
        """
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)

        for key, value in self.__dict__.items():
            if isinstance(value, Experiment):
                setattr(obj, key, value.clone())
            elif isinstance(value, list) and all(isinstance(element, Experiment) for element in value):
                setattr(obj, key, [element.clone() for element in value])

        return obj

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Experiment:
        """Perform the experiment."""
        pass

    def __str__(self):
        """
        Get a nice string representation, which includes nested experiments.
        Note that it explicitly does not include newlines and indentation, because
        getting this right is a whole another problem.
        Just look at `__repr__()` of `sklearn.base.BaseEstimator`.
        """

        if hasattr(self, 'experiments'):
            inner = getattr(self, 'experiments')
        elif hasattr(self, 'experiment'):
            inner = getattr(self, 'experiment')
        else:
            inner = None

        return f"{self.__class__.__name__}({inner})"

    def __repr__(self):
        return str(self)


class Evaluation(Experiment):
    """This experiment evaluates a single model using cross-validation."""

    score_: float

    def __init__(self,
                 estimator: BaseEstimator = None,
                 X_train: np.ndarray = None,
                 X_test: np.ndarray = None,
                 y_train: np.ndarray = None,
                 y_test: np.ndarray = None,
                 random_state: int = None,
                 ):
        self.estimator = clone(estimator)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.random_state = random_state

    def set_params(self, **params):
        """Parameters of the estimator can be supplied using the `estimator__` prefix."""

        super().set_params(**params)

        for key, value in params.items():
            if key.startswith('estimator__'):
                modified_key = key.removeprefix('estimator__')
                self.estimator.set_params(**{modified_key: value})

    def clone(self) -> Experiment:
        """Clone the experiment and the estimator."""
        obj = super().clone()
        obj.estimator = clone(self.estimator)
        return obj

    def __call__(self, *args, **kwargs) -> Evaluation:
        print(f"Starting fit of {self.estimator}")

        self.estimator.fit(self.X_train, self.y_train)
        self.score_ = self.estimator.score(self.X_test, self.y_test)

        # TODO: implement real cv

        print(f"Ended fit, score was {self.score_}")

        return self

    def __str__(self):
        return f"{self.__class__.__name__}({self.estimator})"


class ParameterTuning(Experiment):
    """Perform parameter tuning on the supplied `parameter_space`."""

    def __init__(self, experiment: Experiment, tuner: ParameterTuner = None, parameter_space: dict = None,
                 random_state: int = None):
        self.experiment = experiment
        self.tuner = tuner
        self.parameter_space = parameter_space
        self.random_state = random_state

    def set_params(self, **params):
        super().set_params(**params)
        self.experiment.set_params(**params)

    def __call__(self, *args, **kwargs) -> ParameterTuning:
        # TODO: implement real tuning

        self.experiment(*args, **kwargs)

        return self


class ParameterList(Experiment):
    """
    Extend the supplied experiment using the parameter list.
    A new nested experiment is created for every element in `parameter_list`
    and the `parameter_name` is set to this element.
    """

    def __init__(self, experiment: Experiment,
                 parameter_name: str = None,
                 parameter_list: Union[Iterator, Generator, list, tuple] = None):
        self.experiment = experiment
        self.parameter_name = parameter_name
        self.parameter_list = parameter_list

        self.experiments = []
        for parameter in parameter_list:
            cloned_experiment = self.experiment.clone()
            cloned_experiment.set_params(**{self.parameter_name: parameter})
            self.experiments.append(cloned_experiment)

    def set_params(self, **params):
        super().set_params(**params)

        for experiment in self.experiments:
            experiment.set_params(**params)

    def __call__(self, *args, **kwargs) -> Experiment:
        for experiment in self.experiments:
            experiment(*args, **kwargs)

        return self


class NestedExperiment(Experiment):
    """Perform nested experiments, e.g., for multiple models."""

    def __init__(self, experiments: list[Experiment]):
        self.experiments = experiments

    def set_params(self, **params):
        super().set_params(**params)

        for experiment in self.experiments:
            experiment.set_params(**params)

    def __call__(self, *args, **kwargs) -> Experiment:
        for experiment in self.experiments:
            experiment(*args, **kwargs)

        return self


class FixedParameters(Experiment):
    """Overwrite the parameters of the nested experiments, if the value is None."""

    def __init__(self, experiment: Experiment, parameters: dict = None):
        self.experiment = experiment
        self.parameters = parameters

        self.experiment.set_params(**self.parameters)

    def __call__(self, *args, **kwargs) -> Experiment:
        self.experiment(*args, **kwargs)

        return self
