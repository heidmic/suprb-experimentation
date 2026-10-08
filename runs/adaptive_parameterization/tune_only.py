import os

import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle

from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.multi_objective import MOLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import nsga2, nsga3, spea2
from suprb.optimizer.rule import es, origin, mutation, ns
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.novelty_search_type import MinimalCriteria
from suprb.rule.subsumption import PreferSmallerVolume, PreferLargerVolume
from suprb.solution.initialization import RandomInit

random_state = 42

opt_dict = {
    "nsga2": nsga2.NonDominatedSortingGeneticAlgorithm2,
    "nsga3": nsga3.NonDominatedSortingGeneticAlgorithm3,
    "spea2": spea2.StrengthParetoEvolutionaryAlgorithm2,
}


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets

    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def get_storage_url() -> str:
    url = os.environ.get("OPTUNA_STORAGE")
    if url:
        return url

    for env_var in ("PG_SOCKET_DIR", "SCRATCH", "DEVENV_STATE"):
        socket_dir = os.environ.get(env_var)
        if socket_dir:
            if env_var in ("SCRATCH", "DEVENV_STATE"):
                socket_dir = f"{socket_dir}/postgres"
            return f"postgresql+psycopg2:///optuna_db?host={socket_dir}"

    return "postgresql+psycopg2://localhost/optuna_db"


@click.command()
@click.option("-p", "--problem", type=click.STRING, default="airfoil_self_noise")
@click.option("-j", "--job_id", type=click.STRING, default="NA")
@click.option("-o", "--optimizer", type=click.STRING, default="nsga2")
@click.option("--worker-id", type=click.INT, default=0)
def run(problem: str, job_id: str, optimizer: str, worker_id: int):
    print(f"[tune] Problem: {problem} | optimizer: {optimizer} | job: {job_id} | worker: {worker_id}")

    worker_random_state = random_state + (worker_id * 400)

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=worker_random_state)

    estimator = SupRB(
        rule_discovery=es.ES1xLambda(
            operator="&",
            n_iter=1000,
            delay=30,
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.VolumeWu(),
                model=Ridge(alpha=0.01, random_state=worker_random_state),
            ),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),
            # subsumption=PreferLargerVolume(tolerance=0.01),
        ),
        solution_composition=opt_dict[optimizer](n_iter=32, population_size=32),
        n_iter=32,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", MOLogger())]),
        random_state=worker_random_state,
        early_stopping_patience=5,
        early_stopping_delta=0.0025,
        extra_rules_patience=1,
        extra_rules_delta=0.0025,
        convergence_metric="hypervolume",
    )

    storage_url = get_storage_url()
    study_name = os.environ.get("OPTUNA_STUDY_NAME", f"{optimizer}_tuning_{problem}_job{job_id}")

    trials_per_worker = 4000
    timeout_hours = float(os.environ.get("TIMEOUT_HOURS", 24))
    timeout_seconds = int(timeout_hours * 3600)

    print(f"[tune] Storage : {storage_url}")
    print(f"[tune] Study   : {study_name}")
    print(f"[tune] Trials  : {trials_per_worker}  |  Timeout: {timeout_seconds}s")

    tuning_params = dict(
        estimator=estimator,
        random_state=worker_random_state,
        cv=4,
        n_jobs_cv=1,
        n_jobs=1,
        n_calls=trials_per_worker,
        timeout=timeout_seconds,
        scoring="test_hypervolume",
        verbose=10,
        study_name=study_name,
        storage=storage_url,
    )

    @param_space()
    def suprb_ES_NSGA2_space(trial: Trial, params: Bunch):
        sigma_space = [0, np.sqrt(X.shape[1])]
        params.rule_discovery__mutation__sigma = trial.suggest_float("rule_discovery__mutation__sigma", *sigma_space)
        params.rule_discovery__init__fitness__alpha = trial.suggest_float("rule_discovery__init__fitness__alpha", 0.01, 0.2)
        params.solution_composition__crossover = trial.suggest_categorical("solution_composition__crossover", ["NPoint", "Uniform"])
        params.solution_composition__crossover = getattr(nsga2.crossover, params.solution_composition__crossover)()
        if isinstance(params.solution_composition__crossover, nsga2.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int("solution_composition__crossover__n", 1, 10)
        params.solution_composition__mutation__mutation_rate = trial.suggest_float("solution_composition__mutation_rate", 0, 0.1)

    @param_space()
    def suprb_ES_NSGA3_space(trial: Trial, params: Bunch):
        sigma_space = [0, np.sqrt(X.shape[1])]
        params.rule_discovery__mutation__sigma = trial.suggest_float("rule_discovery__mutation__sigma", *sigma_space)
        params.rule_discovery__init__fitness__alpha = trial.suggest_float("rule_discovery__init__fitness__alpha", 0.01, 0.2)
        params.solution_composition__crossover = trial.suggest_categorical("solution_composition__crossover", ["NPoint", "Uniform"])
        params.solution_composition__crossover = getattr(nsga3.crossover, params.solution_composition__crossover)()
        if isinstance(params.solution_composition__crossover, nsga3.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int("solution_composition__crossover__n", 1, 10)
        params.solution_composition__mutation__mutation_rate = trial.suggest_float("solution_composition__mutation_rate", 0, 0.1)

    @param_space()
    def suprb_ES_SPEA2_space(trial: Trial, params: Bunch):
        sigma_space = [0, np.sqrt(X.shape[1])]
        params.rule_discovery__mutation__sigma = trial.suggest_float("rule_discovery__mutation__sigma", *sigma_space)
        params.rule_discovery__init__fitness__alpha = trial.suggest_float("rule_discovery__init__fitness__alpha", 0.01, 0.2)
        params.solution_composition__crossover = trial.suggest_categorical("solution_composition__crossover", ["NPoint", "Uniform"])
        params.solution_composition__crossover = getattr(spea2.crossover, params.solution_composition__crossover)()
        if isinstance(params.solution_composition__crossover, spea2.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int("solution_composition__crossover__n", 1, 10)
        params.solution_composition__mutation__mutation_rate = trial.suggest_float("solution_composition__mutation_rate", 0, 0.1)

    space_dict = {
        "nsga2": suprb_ES_NSGA2_space,
        "nsga3": suprb_ES_NSGA3_space,
        "spea2": suprb_ES_SPEA2_space,
    }

    experiment_name = f"Baseline {optimizer} j:{job_id} p:{problem}"
    experiment = Experiment(name=experiment_name, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(space_dict[optimizer], tuner=tuner)

    # No evaluation here -- this worker only contributes trials to the
    # shared study. Passing evaluation=None makes Experiment.perform() skip
    # the evaluation branch entirely after tuning finishes.
    experiment.perform(evaluation=None)

    print(f"[tune] Worker {worker_id}: finished contributing trials.")


if __name__ == "__main__":
    run()
