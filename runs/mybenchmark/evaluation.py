from __future__ import annotations
from typing import Optional
from numbers import Integral

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, ShuffleSplit, cross_validate



def _check_scoring(extra: Optional[str | list[str]] = None) -> list[str]:
    base = {"r2", "neg_mean_squared_error", "neg_mean_absolute_error"}
    if extra is not None:
        base.update([extra] if isinstance(extra, str) else extra)
    return list(base)

 

def run_evaluation(
    estimator: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    tuned_params: Optional[dict] = None,
    cv: Optional[ShuffleSplit | KFold] = None,
    n_jobs: int = 1,
    random_state: Optional[int] = None,
    scoring: Optional[str | list[str]] = None,
    verbose: int = 0,
) -> tuple[list[BaseEstimator], dict]:
    
    scoring_list = _check_scoring(scoring)
        
    est = clone(estimator)
    if random_state is not None:
        est.set_params(random_state=random_state)
    if tuned_params:
        est.set_params(**tuned_params)


    print(f"[evaluation] Start cross_validate | cv={cv} | scoring={scoring_list} | n_jobs={n_jobs}")

    raw_scores = cross_validate(
        estimator=est,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring_list,
        n_jobs=n_jobs,
        return_estimator=True,
        verbose=verbose,
        error_score="raise",
    )

    estimators: list[BaseEstimator] = raw_scores.pop("estimator") 
    scores: dict = dict(raw_scores) 

    # print summarized scores 
    for key, arr in scores.items():
        if hasattr(arr, "__len__"):
            print(
                f"[evaluation]  {key}: mean={np.mean(arr):.4f}  std={np.std(arr):.4f}"
            )

    return estimators, scores
