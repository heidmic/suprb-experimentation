import warnings
from typing import Callable, Optional
 
import numpy as np
import optuna
from optuna import Trial

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, cross_validate
from sklearn.utils import Bunch

import suprb
from suprb import rule
from suprb.optimizer.solution import ga

optuna.logging.set_verbosity(optuna.logging.WARNING)

#---------------------------------------------------------------------
# Parameter Space
#---------------------------------------------------------------------
def suprb_param_space(trial: Trial, X: np.ndarray) -> dict:
    sigma_space = (0.0, float(np.sqrt(X.shape[1])))

    crossover_name = trial.suggest_categorical( 
        "solution_composition__crossover", ["NPoint", "Uniform"]
    )
 
    params: dict = {
        # ES
        "rule_discovery__mutation__sigma": trial.suggest_float(
            "rule_discovery__mutation__sigma", *sigma_space
        ),
        "rule_discovery__init__fitness__alpha": trial.suggest_float(
            "rule_discovery__init__fitness__alpha", 0.01, 0.2
        ),
        # GA
        "solution_composition__selection__k": trial.suggest_int(
            "solution_composition__selection__k", 3, 10
        ),
        "solution_composition__mutation_rate": trial.suggest_float(
            "solution_composition__mutation_rate", 0.0, 0.1
        ),
        "solution_composition__crossover": getattr(ga.crossover, crossover_name)(),
    }
 
    if crossover_name == "NPoint":
        params["solution_composition__crossover__n"] = trial.suggest_int(
            "solution_composition__crossover__n", 1, 10
        )
 
    return params

def suprb_param_space_ns(
    trial: Trial,
    X: np.ndarray,
    ns_type: str = "MCNS",
    use_current_population: bool = False,
) -> dict:
    params = Bunch()

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

    params.rule_discovery__init = trial.suggest_categorical(
        "init", ["MeanInit", "NormalInit", "HalfnormInit"]
    )
    params.rule_discovery__init = getattr(rule.initialization, params.rule_discovery__init)()

    params.rule_discovery__selection = trial.suggest_categorical(
        "selection", ["RouletteWheel", "Random"]
    )
    params.rule_discovery__selection = getattr(
        suprb.optimizer.rule.selection, params.rule_discovery__selection
    )()

    params.rule_discovery__mutation = trial.suggest_categorical('mutation',
        ['Normal', 'Halfnorm','HalfnormIncrease', 'Uniform','UniformIncrease', ])
    params.rule_discovery__mutation = getattr(
        suprb.optimizer.rule.mutation, params.rule_discovery__mutation)()
    
    if ns_type is None:
        novelty_search_type_name = trial.suggest_categorical(
            "novelty_search_type", ["NoveltySearchType", "MinimalCriteria", "LocalCompetition"]
        )
    elif ns_type.upper() == "NS":
        novelty_search_type_name = "NoveltySearchType"
    elif ns_type.upper() == "MCNS":
        novelty_search_type_name = "MinimalCriteria"
    elif ns_type.upper() == "NSLC":
        novelty_search_type_name = "LocalCompetition"
    else:
        raise ValueError(f"Unbekannter ns_type: {ns_type}")

    params.rule_discovery__novelty_calculation__novelty_search_type = getattr(
        suprb.optimizer.rule.ns.novelty_search_type, novelty_search_type_name
    )()

    if isinstance(
        params.rule_discovery__novelty_calculation__novelty_search_type,
        suprb.optimizer.rule.ns.novelty_search_type.MinimalCriteria,
    ):
        params.rule_discovery__novelty_calculation__novelty_search_type__min_examples_matched = (
            trial.suggest_int("min_examples_matched", 5, 15)
        )
    elif isinstance(
        params.rule_discovery__novelty_calculation__novelty_search_type,
        suprb.optimizer.rule.ns.novelty_search_type.LocalCompetition,
    ):
        params.rule_discovery__novelty_calculation__novelty_search_type__max_neighborhood_range = (
            trial.suggest_int("max_neighborhood_range", 10, 20)
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
        params.rule_discovery__novelty_calculation,
        suprb.optimizer.rule.ns.novelty_calculation.NoveltyFitnessBiased,
    ):
        params.rule_discovery__novelty_calculation__k_neighbor = trial.suggest_int("k_neighbor", 10, 20)
    else:
        params.rule_discovery__novelty_calculation__novelty_bias = trial.suggest_float("novelty_bias", 0.3, 0.7)

    # from cli
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

    params.solution_composition__mutation_rate = trial.suggest_float(
        "solution_composition__mutation_rate", 0.0, 0.1
    )

    return dict(params)

#---------------------------------------------------------------------
# Tuning
#---------------------------------------------------------------------
def run_tuning(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    param_space_fn: Callable[[Trial, np.ndarray], dict],
    study_name: str,
    storage_url,   
    n_trials: int = 200,
    timeout: Optional[float] = None,
    cv: int = 4,
    n_jobs_cv: int = 1,
    n_jobs: int = 1,    # if sqlte, n_jobs must be 1
    random_state: int = 42,
    scoring: str = "neg_mean_squared_error",
    verbose: int = 0,

) -> dict:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        sampler = optuna.samplers.TPESampler(seed=random_state)

        # Falls die Study schon existiert: löschen, statt fortzusetzen
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
            print(f"[tuning] Existing study '{study_name}' found and deleted, starting fresh.")
        except KeyError:
            pass  # Study existierte noch nicht -> nichts zu tun

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=False,
            direction="minimize",
            sampler=sampler,
        )

        def objective(trial: Trial) -> float:
            params = param_space_fn(trial, X)
            est = clone(estimator)
            est.set_params(**params)
            try:
                scores = cross_validate(
                    estimator=est,
                    X=X,
                    y=y,
                    cv=cv_splitter,
                    scoring=scoring,
                    n_jobs=n_jobs_cv,
                    return_estimator=False,
                    error_score="raise",
                )
                return -float(np.mean(scores["test_score"]))
            except Exception as exc:
                warnings.warn(f"[tuning] Trial {trial.number} failed: {exc}")
                return float("inf")

        study.optimize(
            func=objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar= bool(verbose),
        )

        print(f"[tuning] Best Values: {study.best_value:.6f}")
        print(f"[tuning] Best Trial-Params (Optuna): {study.best_params}")

    
        best_params = param_space_fn(study.best_trial, X)
        print(f"[tuning] Best set_params-kompatible Params: {best_params}")
        
        return best_params

