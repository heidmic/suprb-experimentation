import sys

import numpy as np
import click
import mlflow
from optuna import Trial

import scipy.stats as stats
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate, MOOCrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.multi_objective import MOLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga, nsga2, nsga3, spea2, ts
from suprb.optimizer.solution.base import MOSolutionComposition
from suprb.optimizer.rule import es
from suprb.optimizer.solution.sampler import BetaSolutionSampler, DiversitySolutionSampler


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets

    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option("-p", "--problem", type=click.STRING, default="airfoil_self_noise")
@click.option("-j", "--job_id", type=click.STRING, default="NA")
@click.option("-c", "--config", type=click.STRING, default="ga32")
def run(problem: str, job_id: str, config: str):
    print(f"Problem is {problem}, with job id {job_id} and optimizer GA")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    config_dict = {
        "ga32": ts.TwoStageSolutionComposition(
            algorithm_1=ga.GeneticAlgorithm(),
        ),
        "ga64": ts.TwoStageSolutionComposition(
            algorithm_1=ga.GeneticAlgorithm(),
            algorithm_2=ga.GeneticAlgorithm(n_iter=64),
            switch_iteration=32,
        ),
        "ga_no_tuning": ts.TwoStageSolutionComposition(
            algorithm_1=ga.GeneticAlgorithm(),
        ),
    }

    estimator = SupRB(
        rule_discovery=es.ES1xLambda(
            operator="&",
            n_iter=1000,
            delay=30,
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.VolumeWu(), model=Ridge(alpha=0.01, random_state=random_state)
            ),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),
        ),
        solution_composition=config_dict[config],
        n_iter=32,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", MOLogger())]),
        random_state=random_state,
    )

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=1000,
        timeout=60 * 60 * 24 * 3 if not sys.gettrace() else 60,
        scoring="test_elitist_fitness",
        verbose=10,
    )

    @param_space()
    def suprb_ES_MOO_space(trial: Trial, params: Bunch):
        # ES
        sigma_space = [0, np.sqrt(X.shape[1])]

        params.rule_discovery__mutation__sigma = trial.suggest_float("rule_discovery__mutation__sigma", *sigma_space)
        params.rule_discovery__init__fitness__alpha = trial.suggest_float(
            "rule_discovery__init__fitness__alpha", 0.01, 0.2
        )

        ############ Solution Composition ##############
        ############ GA First Stage ##############
        params.solution_composition__algorithm_1__selection__k = trial.suggest_int(
            "solution_composition__selection__k", 3, 10
        )

        params.solution_composition__algorithm_1__crossover = trial.suggest_categorical(
            "solution_composition_algorithm_1__crossover", ["NPoint", "Uniform"]
        )
        params.solution_composition__algorithm_1__crossover = getattr(
            ga.crossover, params.solution_composition__algorithm_1__crossover
        )()

        if isinstance(params.solution_composition__algorithm_1__crossover, ga.crossover.NPoint):
            params.solution_composition__algorithm_1__crossover__n = trial.suggest_int(
                "solution_composition__crossover__n", 1, 10
            )

        params.solution_composition__algorithm_1__mutation_rate = trial.suggest_float(
            "solution_composition_algorithm_1__mutation_rate", 0, 0.1
        )

        ######################## Second Stage ##########################
        if isinstance(estimator.solution_composition.algorithm_2, ga.GeneticAlgorithm):
            params.solution_composition__algorithm_2__selection__k = (
                params.solution_composition__algorithm_1__selection__k
            )
            params.solution_composition__algorithm_2__crossover = params.solution_composition__algorithm_1__crossover
            if isinstance(params.solution_composition__algorithm_2__crossover, ga.crossover.NPoint):
                params.solution_composition__algorithm_2__crossover__n = (
                    params.solution_composition__algorithm_1__crossover__n
                )
            params.solution_composition__algorithm_2__mutation_rate = (
                params.solution_composition__algorithm_1__mutation_rate
            )

    experiment_name = f"Baseline c:{config} j:{job_id} p:{problem}"
    print(experiment_name)
    experiment = Experiment(name=experiment_name, verbose=10)

    if not config.endswith("_no_tuning"):
        tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
        experiment.with_tuning(suprb_ES_MOO_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    evaluation = MOOCrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == "__main__":
    run()
