from __future__ import annotations

import itertools
from typing import Union, Any

from sklearn.base import BaseEstimator
from sklearn.utils import Bunch

from experiment.evaluation import Evaluation
from experiment.parameter_search import ParameterTuner


def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


class Experiment:
    """Performs an experiment."""

    results_: Bunch
    estimators_: list[BaseEstimator]

    tuned_params_: dict
    tuning_results_: Any

    def __init__(self, name: str = 'Default', params: dict = None, tuner: ParameterTuner = None, verbose: int = 1):
        self.name = name
        self.tuner = tuner
        self.verbose = verbose

        self.params = params if params is not None else {}
        self.param_space = {}
        self.experiments = []

    def perform(self, evaluation: Evaluation, **kwargs):
        """Perform the experiment."""

        # Tune, if both the parameter space and the tuner are set
        if self.param_space and self.tuner is not None:
            self.log("Starting parameter tuning", reason='tuning', fill='-')
            self.log(f"Parameter space is {self.param_space}", reason='tuning', priority=5)
            tuned_params, tuning_result = self.tuner(parameter_space=self.param_space)
            self.tuned_params_ = tuned_params
            self.tuning_results_ = tuning_result

            self.log(f"Ended parameter tuning", reason='tuning', fill='-')
            self.log(f"Results were {tuned_params}", reason='tuning', priority=5)

            # Propagate to nested experiments
            for experiment in self.experiments:
                experiment.params |= tuned_params
        else:
            tuned_params = {}

        # Evaluate either itself or call on nested experiments
        if self.experiments:
            for experiment in self.experiments:
                experiment.perform(evaluation=evaluation, **kwargs)
        else:
            self.log("Starting evaluation", reason='eval', fill='-')
            params = self.params | tuned_params
            self.estimators_, result = evaluation(params=params, **kwargs)
            self.results_ = Bunch(**result)
            self.log("Ended evaluation", reason='eval', fill='-')

    def with_params(self, params: dict) -> Union[Experiment, list[Experiment]]:
        """
        Return a new nested experiment with the parameters supplied.
        :param params: The parameter dict. If any of the parameter values is a list,
         the parameters will be expanded to all combinations of parameters.
        :return: All nested experiments, either directly or as list, depending on params`.
        """

        # Perform product of all parameters, if necessary
        if any(map(lambda value: isinstance(value, list), params.values())):
            params = {key: list(value) if not isinstance(value, list) else value for key, value in params.items()}
            param_list = product_dict(**params)
        else:
            param_list = [params]

        # Create a new experiment for every parameter combination
        experiments = []
        for params in param_list:
            experiment = self._clone()
            experiment.params |= params
            experiments.append(experiment)

        self.experiments.extend(experiments)

        if len(experiments) > 1:
            return experiments
        else:
            return experiments[0]

    def with_tuning(self, param_space: dict, tuner: ParameterTuner = None, propagate: bool = False) -> Experiment:
        """
        Add parameter tuning to the experiment.
        :param param_space: The parameter space which should be optimized.
        :param tuner: The tuning method. Can overwrite the tuner set by super-experiments.
        :param propagate: If the tuning should be propagated to nested experiments. The parameter spaces are
        merged, not overwritten. Note that this is only done in retrospective, so nested experiments added after
        this method call will not be influenced.
        :return:
        """

        if not propagate or not self.experiments:
            self.param_space |= param_space
            if tuner is not None:
                self.tuner = tuner
        else:
            for experiment in self.experiments:
                experiment.with_tuning(param_space=param_space, tuner=tuner, propagate=True)

        return self

    def _clone(self) -> Experiment:
        return Experiment(params=self.params.copy() if self.params else None, tuner=self.tuner, verbose=self.verbose)

    def log(self, message: str, reason: str, fill: str = None, priority=1):
        if self.verbose >= priority:
            message = f"[{reason.upper()}] {message}"
            if fill is not None:
                message = f"{message} ".ljust(80, fill)
            print(message, flush=True)

    def __str__(self):
        return f"<Experiment:{self.name};nested={len(self.experiments)};depth={self._height}>"

    def __repr__(self):
        return str(self)

    @property
    def _height(self) -> int:
        """Calculates the height in the nested experiment tree, i.e., leaves have a height of zero."""
        if not self.experiments:
            return 0
        else:
            return max(map(lambda ex: ex._height, self.experiments)) + 1
