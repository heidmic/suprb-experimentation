from __future__ import annotations
import numpy as np
import click
import time
import mlflow

from datetime import datetime

from optuna import Trial
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from problems import scale_X_y

from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.spea2 import StrengthParetoEvolutionaryAlgorithm2
from suprb.optimizer.solution.sampler import BetaSolutionSampler
from suprb.logging.multi_objective import MOLogger

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es, origin, mutation, ns
from suprb.solution.initialization import RandomInit

from evaluation import run_evaluation
from mlflow_logging import log_results, log_results_multi_seed
from tuning import run_tuning, suprb_param_space

import os
from joblib import Parallel, delayed


""" # Zugriff auf Multi-Objective-Ergebnisse:
logger = model.logger_
pareto_front = logger.pareto_fronts_[-1]      # letzte Iteration
hv = logger.metrics_["hypervolume"][-1]
spread = logger.metrics_["spread"][-1] """


RANDOM_STATE = 42
N_CPU = int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) 

_HERE = os.path.dirname(os.path.abspath(__file__))

MLFLOW_URI    = os.path.join(_HERE, "mlruns")
OPTUNA_DB_DIR = os.path.join(_HERE, "optuna_dbs")
os.makedirs(OPTUNA_DB_DIR, exist_ok=True)

def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)
    raise ValueError(f"Kein Dataset '{name}' gefunden (erwartet: problems.datasets.load_{name})")

    
def build_estimator(n_iter: int, n_rules: int, n_initial_rules: int) -> SupRB:

    model = SupRB(
        rule_discovery=ES1xLambda(n_iter=1000, delay=30), #TODO: oder ES1xLambda wie bei deafault? 

        solution_composition=StrengthParetoEvolutionaryAlgorithm2(
            n_iter=32,
            population_size=32,
            sampler=BetaSolutionSampler(),
            early_stopping_delta=0.01,   # min. geforderte Hypervolumen-Verbesserung
            early_stopping_patience=10,  # Iterationen ohne Verbesserung, bevor abgebrochen wird
        ),
        logger=MOLogger(),                # mologger includes default logger                                  
        n_iter=n_iter,
        n_rules=n_rules,
        n_initial_rules=n_initial_rules,
        verbose=1,
    )
    return model
    
    """ SupRB(
        rule_discovery=es.ES1xLambda(
            operator="&",
            n_iter=1000,
            delay=30,
            init=rule.initialization.MeanInit(
                fitness=rule.fitness.VolumeWu(),
                model=Ridge(alpha=0.01, random_state=RANDOM_STATE),
            ),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),
        ),

        solution_composition=ga.GeneticAlgorithm(
            n_iter=32,
            population_size=32,
            selection=ga.selection.Tournament(),
        ),

        n_iter=n_iter,
        n_rules=n_rules,
        n_initial_rules=n_initial_rules,
        verbose=1, 
        logger=CombinedLogger(
            [("stdout", StdoutLogger()), ("default", DefaultLogger())]
        ),
    ) """

def _evaluate_one_seed(estimator, X, y, tuned_params, rs):
        cv_splitter = ShuffleSplit(n_splits=30, test_size=0.25, random_state=int(rs))
        print(f"[evaluation] [{datetime.now():%Y-%m-%d %H:%M:%S}] Seed {rs} gestartet", flush=True)
        result = run_evaluation(
            estimator=estimator,
            X=X, y=y,
            tuned_params=tuned_params,
            cv=cv_splitter,
            n_jobs=1,          
            random_state=int(rs),
            verbose=2, 
        )
        print(f"[evaluation] [{datetime.now():%Y-%m-%d %H:%M:%S}] Seed {rs} abgeschlossen", flush=True)
        return result


@click.command()
@click.option("-p", "--problem",          type=str, default="airfoil_self_noise", show_default=True)
@click.option("-j", "--job_id",           type=str, default="NA",                 show_default=True)
@click.option("-n", "--n_iter",           type=int, default=32,                   show_default=True)
@click.option("-r", "--n_rules",          type=int, default=4,                    show_default=True)
@click.option("-i", "--n_initial_rules",  type=int, default=4,                    show_default=True)
#@click.option("--tune/--no-tune",                   default=True,                 show_default=True)
def run(
    problem: str,
    job_id: str,
    n_iter: int,
    n_rules: int,
    n_initial_rules: int,
    #tune: bool,
):

    print(f"[run] Problem={problem}  job_id={job_id}  n_iter={n_iter}  "
        f"n_rules={n_rules}  n_initial_rules={n_initial_rules} MOO")

    #tune_label = "tune" if tune else "notune"
    label = "moo"
    
    t0 = time.perf_counter()

    #-----------------------------------------------------------------------
    # Data
    #-----------------------------------------------------------------------

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y, _ = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=RANDOM_STATE)

    #-----------------------------------------------------------------------
    # Estimator
    #-----------------------------------------------------------------------

    grid_params = dict(n_iter=n_iter, n_rules=n_rules, n_initial_rules=n_initial_rules)
    estimator = build_estimator(**grid_params)

    #-------------------------------------------------------------------------
    # Optinal Tuning (optional)
    # trails sequentiell, cv parallel (innerhalb jedes trials)
    #--------------------------------------------------------------------------
    tuned_params: dict = {}
    tuning_walltime: float = 0.0

    study_name = f"{problem}__ni{n_iter}__nr{n_rules}__nir{n_initial_rules}__{label}"
    
    """  if tune: 
        sub_dir = f"{problem}"
        os.makedirs(os.path.join(OPTUNA_DB_DIR, sub_dir), exist_ok=True)
        db_url = f"sqlite:///{OPTUNA_DB_DIR}/{sub_dir}/{study_name}.db" #keine gemeinsame DB der SLURM jobs, da parallele Optuna-Trials zu Konflikten führen würden. Stattdessen: separate DB pro Job/Studie.

        print(f"Starting tuning for {study_name}")
        tuned_params = run_tuning(
            estimator=estimator,
            X=X,
            y=y,
            param_space_fn=suprb_param_space,
            study_name=study_name,
            storage_url=db_url,
            n_trials=1000,
            timeout=60*60*24,  # 24 hours
            cv=4,
            n_jobs_cv=N_CPU, #parallelität innerhalb cv jedes trials 
            n_jobs=1, #prallelität der trials, sqlite -> n_jobs=1
            random_state=RANDOM_STATE,
            scoring="neg_mean_squared_error",
            verbose=1,
        )

        tuning_walltime = time.perf_counter() - t0
        print(f"[tuning] Tuning completed in {tuning_walltime:.2f} seconds")

        print(f"[tuning] Best Params: {tuned_params}") """
    
    #-------------------------------------------------------------------------
    # Evaluation
    # Seeds parallel, cv sequentiell
    #--------------------------------------------------------------------------
    t1 = time.perf_counter()

    random_states = np.random.SeedSequence(RANDOM_STATE).generate_state(8)
    
    seed_results: list[tuple] = []

    seed_results = Parallel(n_jobs=N_CPU)(
        delayed(_evaluate_one_seed)(estimator, X, y, tuned_params, rs)
        for rs in random_states
    )

    evaluation_walltime = time.perf_counter() - t1
    print(f"[evaluation] [{datetime.now():%Y-%m-%d %H:%M:%S}] Evaluation completed in {evaluation_walltime:.2f} seconds")
    
    walltime_metrics = dict(
        tuning_walltime_s=round(tuning_walltime, 2),
        evaluation_walltime_s=round(evaluation_walltime, 2),
    )



    #-------------------------------------------------------------------------
    # MLflow Logging
    #--------------------------------------------------------------------------
    
    experiment_name = f"SupRB | problem={problem} | {label}" 
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)

    log_results_multi_seed(
        run_name=study_name,
        grid_params=grid_params,
        tuned_params=tuned_params,
        seed_results=seed_results,          # liste der (estimators, cv_results)-Tupel
        random_states=random_states,
        walltime_metrics=walltime_metrics,
    )
    print(f"[run] MLflow-Ergebnisse geloggt unter Experiment '{experiment_name}'")


if __name__ == "__main__":
    run()



