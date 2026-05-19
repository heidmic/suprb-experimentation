import numpy as np
import click
import mlflow
from optuna import Trial
import suprb

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import ns, origin
from suprb.optimizer.rule.mutation import HalfnormIncrease
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.archive import ArchiveNovel
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd

random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets

    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)
    else:
        enc = LabelEncoder()
        dataset = fetch_openml(name=name, version=1)

        if name == "meta":
            dataset.data.DS_Name = enc.fit_transform(dataset.data.DS_Name)
            dataset.data.Alg_Name = enc.fit_transform(dataset.data.Alg_Name)
            dataset.data = dataset.data.drop(dataset.data.columns[dataset.data.isna().any()].tolist(), axis=1)

        if name == "chscase_foot":
            dataset.data.col_1 = enc.fit_transform(dataset.data.col_1)

        if isinstance(dataset.data, np.ndarray):
            X = dataset.data
        elif isinstance(dataset.data, pd.DataFrame) or isinstance(dataset.data, pd.Series):
            X = dataset.data.to_numpy(dtype=float)
        else:
            X = dataset.data.toarray()

        if isinstance(dataset.target, np.ndarray):
            y = dataset.target
        elif isinstance(dataset.target, pd.DataFrame) or isinstance(dataset.target, pd.Series):
            y = dataset.target.to_numpy(dtype=float)
        else:
            y = dataset.target.toarray()

        return X, y


@click.command()
@click.option("-p", "--problem", type=click.STRING, default="airfoil_self_noise")
@click.option("-t", "--ns_type", type=click.STRING, default=None)
@click.option("-a", "--use_current_population", type=click.BOOL, default=False)
@click.option("-i", "--job_id", type=click.INT, default=None)
def run(problem: str, ns_type: str, use_current_population: bool, job_id: int):
    print(f"{ns_type} with use_current_population={use_current_population} is tuned and tested with problem {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_discovery=ns.NoveltySearch(
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.VolumeWu(), model=Ridge(alpha=0.01, random_state=random_state)
            ),
            origin_generation=origin.SquaredError(),
            mutation=HalfnormIncrease(),
        ),
        solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32),
        n_iter=32,
        n_rules=8,
        verbose=10,
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
    )

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=10_000,
        timeout=72 * 60 * 60,  # 72 hours
        scoring="fitness",
        verbose=10,
    )

    @param_space()
    def suprb_NS_GA_space(trial: Trial, params: Bunch):
        # NS
        # sigma_space = [0, 2]
        sigma_space = [0, np.sqrt(X.shape[1])]
        params.rule_discovery__mutation__sigma = trial.suggest_float("mutation_sigma", *sigma_space)

        params.rule_discovery__n_iter = trial.suggest_int("n_iter", 0, 20)
        params.rule_discovery__mu = trial.suggest_int("mu", 7, 20)
        params.rule_discovery__lmbda = trial.suggest_int("lmbda", 28, 200)
        params.rule_discovery__roh = trial.suggest_int("roh", 10, 75)

        params.rule_discovery__origin_generation = trial.suggest_categorical(
            "origin_generation", ["UniformSamplesOrigin", "Matching", "SquaredError"]
        )
        params.rule_discovery__origin_generation = getattr(
            suprb.optimizer.rule.origin, params.rule_discovery__origin_generation
        )()

        params.rule_discovery__init = trial.suggest_categorical("init", ["MeanInit", "NormalInit", "HalfnormInit"])
        params.rule_discovery__init = getattr(rule.initialization, params.rule_discovery__init)()

        params.rule_discovery__selection = trial.suggest_categorical("selection", ["RouletteWheel", "Random"])
        params.rule_discovery__selection = getattr(suprb.optimizer.rule.selection, params.rule_discovery__selection)()

        params.rule_discovery__mutation__sigma = trial.suggest_float("mutation_sigma", *sigma_space)
        params.rule_discovery__mutation = trial.suggest_categorical(
            "mutation",
            [
                "Normal",
                "Halfnorm",
                "HalfnormIncrease",
                "Uniform",
                "UniformIncrease",
            ],
        )
        params.rule_discovery__mutation = getattr(suprb.optimizer.rule.mutation, params.rule_discovery__mutation)()

        if ns_type is None:
            params.rule_discovery__novelty_calculation__novelty_search_type = trial.suggest_categorical(
                "novelty_search_type", ["NoveltySearchType", "MinimalCriteria", "LocalCompetition"]
            )
        elif ns_type.upper() == "NS":
            params.rule_discovery__novelty_calculation__novelty_search_type = "NoveltySearchType"
        elif ns_type.upper() == "MCNS":
            params.rule_discovery__novelty_calculation__novelty_search_type = "MinimalCriteria"
        elif ns_type.upper() == "NSLC":
            params.rule_discovery__novelty_calculation__novelty_search_type = "LocalCompetition"

        params.rule_discovery__novelty_calculation__novelty_search_type = getattr(
            suprb.optimizer.rule.ns.novelty_search_type, params.rule_discovery__novelty_calculation__novelty_search_type
        )()

        if isinstance(
            params.rule_discovery__novelty_calculation__novelty_search_type,
            suprb.optimizer.rule.ns.novelty_search_type.MinimalCriteria,
        ):
            params.rule_discovery__novelty_calculation__novelty_search_type__min_examples_matched = trial.suggest_int(
                "min_examples_matched", 5, 15
            )
        elif isinstance(
            params.rule_discovery__novelty_calculation__novelty_search_type,
            suprb.optimizer.rule.ns.novelty_search_type.LocalCompetition,
        ):
            params.rule_discovery__novelty_calculation__novelty_search_type__max_neighborhood_range = trial.suggest_int(
                "max_neighborhood_range", 10, 20
            )

        params.rule_discovery__novelty_calculation__archive = trial.suggest_categorical(
            "archive", ["ArchiveNovel", "ArchiveRandom", "ArchiveNone"]
        )
        params.rule_discovery__novelty_calculation__archive = getattr(
            suprb.optimizer.rule.ns.archive, params.rule_discovery__novelty_calculation__archive
        )()

        params.rule_discovery__novelty_calculation = trial.suggest_categorical(
            "novelty_calculation",
            ["NoveltyCalculation", "ProgressiveMinimalCriteria", "NoveltyFitnessPareto", "NoveltyFitnessBiased"],
        )

        params.rule_discovery__novelty_calculation = getattr(
            suprb.optimizer.rule.ns.novelty_calculation, params.rule_discovery__novelty_calculation
        )()

        if not isinstance(
            params.rule_discovery__novelty_calculation, suprb.optimizer.rule.ns.novelty_calculation.NoveltyFitnessBiased
        ):
            params.rule_discovery__novelty_calculation__k_neighbor = trial.suggest_int("k_neighbor", 10, 20)

        if isinstance(
            params.rule_discovery__novelty_calculation, suprb.optimizer.rule.ns.novelty_calculation.NoveltyFitnessBiased
        ):
            params.rule_discovery__novelty_calculation__novelty_bias = trial.suggest_float("novelty_bias", 0.3, 0.7)

        params.rule_discovery__use_population_for_archive = use_current_population

        # GA
        params.solution_composition__selection = trial.suggest_categorical(
            "solution_composition__selection", ["RouletteWheel", "Tournament", "LinearRank", "Random"]
        )
        params.solution_composition__selection = getattr(ga.selection, params.solution_composition__selection)()

        if isinstance(params.solution_composition__selection, ga.selection.Tournament):
            params.solution_composition__selection__k = trial.suggest_int("solution_composition__selection__k", 3, 10)

        params.solution_composition__crossover = trial.suggest_categorical(
            "solution_composition__crossover", ["NPoint", "Uniform"]
        )
        params.solution_composition__crossover = getattr(ga.crossover, params.solution_composition__crossover)()

        if isinstance(params.solution_composition__crossover, ga.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int("solution_composition__crossover__n", 1, 10)

        params.solution_composition__mutation__mutation_rate = trial.suggest_float(
            "solution_composition__mutation_rate", 0, 0.1
        )

    experiment = Experiment(
        name=(
            f"{problem} {ns_type} {use_current_population} Tuning & Experimentation"
            if job_id is None
            else f"{job_id}: {problem} {ns_type} {use_current_population} Tuning & Experimentation"
        ),
        verbose=10,
    )

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_NS_GA_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(f"{ns_type} Tuning & Experiment")
    log_experiment(experiment)


if __name__ == "__main__":
    run()
