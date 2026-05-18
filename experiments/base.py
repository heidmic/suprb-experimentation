from __future__ import annotations

import itertools
from datetime import datetime
from typing import Union, Any, Optional, Callable

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils import Bunch
import os

from experiments.evaluation import Evaluation
from experiments.parameter_search import ParameterTuner

import suprb.json as suprb_json


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

    def __init__(
        self,
        name: str = "Default",
        params: dict = None,
        tuner: ParameterTuner = None,
        n_jobs: int = None,
        verbose: int = 1,
    ):
        self.name = name
        self.tuner = tuner
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.params = params if params is not None else {}
        self.param_space = {}
        self.experiments = []

    def perform(self, evaluation: Optional[Evaluation], **kwargs) -> Experiment:
        """Perform the experiment."""

        tuning_start = datetime.now()

        # Tune, if both the parameter space and the tuner are set
        if self.param_space and self.tuner is not None:
            self.log(f"Starting parameter tuning at {tuning_start}", reason="tuning", fill="-")
            self.log(f"Parameter space is {self.param_space}", reason="tuning", priority=5)
            tuned_params, tuning_result = self.tuner(parameter_space=self.param_space, local_params=self.params)
            self.tuned_params_ = tuned_params
            self.tuning_results_ = tuning_result

            tuning_stop = datetime.now()
            tuning_delta = tuning_stop - tuning_start

            self.log(f"Ended parameter tuning at {tuning_stop}, took {tuning_delta}", reason="tuning", fill="-")
            self.log(f"Tuning results were {tuned_params}", reason="tuning", priority=5)

            # Propagate to nested experiments
            self._propagate_params(tuned_params)
        else:
            tuned_params = {}

        start = datetime.now()

        # Evaluate either itself or call on nested experiments
        nested = "nested-" if self.experiments else ""
        if evaluation is not None:
            self.log(f"Starting evaluation at {start}", reason=f"{nested}eval", fill="-")
            if self.experiments:
                with Parallel(n_jobs=self.n_jobs) as parallel:
                    self.experiments = parallel(
                        delayed(experiment.perform)(evaluation=evaluation, **kwargs) for experiment in self.experiments
                    )
            else:
                params = self.params | tuned_params
                self.estimators_, result = evaluation(params=params, **kwargs)
                self.results_ = Bunch(**result)
                self.log(f"Evaluation results were {self.results_}", reason=f"{nested}eval", priority=5)

                if not os.path.exists("output_json"):
                    os.makedirs("output_json")

                try:
                    i = 1
                    for score in self.estimators_:
                        suprb_json.dump(score, f"output_json/{self.name}_config_{i}.json")
                        i += 1
                except:
                    print("JSON dump did not work! Continue eval!")

            end = datetime.now()
            delta = end - start
            self.log(f"Ended evaluation at {end}, took {delta}", reason=f"{nested}eval", fill="-")
            total_delta = end - tuning_start
            self.log(f"Total runtime: {total_delta}", reason="stats", priority=5)
        else:
            self.log(f"Skipping evaluation, because None was passed", reason=f"{nested}eval", fill="-")

        return self

    def _propagate_params(self, params: dict):
        """Propagate parameters to all nested experiments."""

        self.params |= params
        for experiment in self.experiments:
            experiment._propagate_params(params)

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

    def with_tuning(
        self,
        param_space: Union[dict, Callable],
        tuner: ParameterTuner = None,
        propagate: bool = False,
        overwrite: bool = False,
    ) -> Experiment:
        """
        Add parameter tuning to the experiment.
        :param param_space: The parameter space which should be optimized.
        :param tuner: The tuning method. Can overwrite the tuner set by super-experiments.
        :param propagate: If the tuning should be propagated to nested experiments. The parameter spaces are
        merged, if possible. Note that this is only done in retrospective, so nested experiments added after
        this method call will not be influenced.
        :param overwrite: Decides if the old parameter space should be overwritten, if merging them is not possible.
        :return:
        """

        if not propagate or not self.experiments:
            if isinstance(self.param_space, dict) and isinstance(param_space, dict):
                self.param_space |= param_space
            elif overwrite or not self.param_space:
                self.param_space = param_space
            if tuner is not None:
                self.tuner = tuner
        else:
            for experiment in self.experiments:
                experiment.with_tuning(param_space=param_space, tuner=tuner, propagate=True)

        return self

    def with_random_states(self, random_states: list[int], n_jobs: int = None) -> Optional[list[Experiment]]:
        """Expand this experiment with copies that use different random states."""

        if not self.experiments:
            new_experiments = []
            for i, random_state in enumerate(random_states):
                new_experiment = self._clone()
                new_experiment.name = f"{self.name}:{i}:{random_state}"
                new_experiment.params |= {"random_state": random_state}
                new_experiments.append(new_experiment)
            self.experiments = new_experiments
            self.n_jobs = n_jobs
            return new_experiments
        else:
            for experiment in self.experiments:
                experiment.with_random_states(random_states)

    def _clone(self) -> Experiment:
        return Experiment(params=self.params.copy() if self.params else None, tuner=self.tuner, verbose=self.verbose)

    def log(self, message: str, reason: str, fill: str = None, priority=1):
        if self.verbose >= priority:
            message = f"[{self.name}] [{reason.upper()}] {message}"
            if fill is not None:
                message = f"{message} ".ljust(80, fill)
            print(message, flush=True)

    def __str__(self):
        return f"<Experiment:{self.name};nested={len(self.experiments)};height={self._height}>"

    def __repr__(self):
        return str(self)

    @property
    def _height(self) -> int:
        """Calculates the height in the nested experiment tree, i.e., leaves have a height of zero."""
        if not self.experiments:
            return 0
        else:
            return max(map(lambda ex: ex._height, self.experiments)) + 1
