import warnings
from typing import Callable, Optional
 
import numpy as np
import optuna
from optuna import Trial
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, cross_validate
 
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

