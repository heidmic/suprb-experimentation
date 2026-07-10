from __future__ import annotations
 
import json
import warnings
from typing import Optional
 
import mlflow
import numpy as np
from sklearn.base import BaseEstimator
from suprb import SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
 
#MLFLOW_ENABLE_ASYNC_LOGGING = True # async logging for speed 
# ---------------------------------------------------------------------------
# Interne Hilfsfunktionen
# ---------------------------------------------------------------------------
 
 
def _get_default_logger(estimator: BaseEstimator) -> Optional[DefaultLogger]:
    if not isinstance(estimator, SupRB):
        return None
    logger = getattr(estimator, "logger_", None)
    if logger is None:
        return None
    if isinstance(logger, DefaultLogger):
        return logger
    if isinstance(logger, CombinedLogger):
        for _, sublogger in logger.loggers_:
            if isinstance(sublogger, DefaultLogger):
                return sublogger
    return None
 
 
def _safe_log_dict(d: dict, artifact_name: str) -> None:
    try:
        mlflow.log_dict(d, artifact_name)
    except TypeError:
        try:
            mlflow.log_text(json.dumps(d, default=str), artifact_name)
        except Exception as exc:
            warnings.warn(f"[mlflow] Konnte '{artifact_name}' nicht loggen: {exc}")
 
 
def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
 
 
""" def _log_estimator_run(estimator: BaseEstimator) -> None:
    try:
        _safe_log_dict(estimator.get_params(), "params.json")
    except Exception as exc:
        warnings.warn(f"[mlflow] Estimator-Params-Log fehlgeschlagen: {exc}")
 
    logger = _get_default_logger(estimator)
    if logger is not None:
        for key, values in logger.metrics_.items():
            for step, value in values.items():
                fval = _safe_float(value)
                if fval is not None:
                    mlflow.log_metric(key=key, value=fval, step=step) """
 
def _log_estimator_run(estimator: BaseEstimator) -> None:
    try:
        _safe_log_dict(estimator.get_params(), "params.json")
    except Exception as exc:
        warnings.warn(f"[mlflow] Estimator-Params-Log fehlgeschlagen: {exc}")

    logger = _get_default_logger(estimator)
    if logger is None:
        return

    by_step: dict[int, dict[str, float]] = {} # pro step: {key: value}
    for key, values in logger.metrics_.items():
        for step, value in values.items():
            fval = _safe_float(value)
            if fval is not None:
                by_step.setdefault(step, {})[key] = fval

    for step, metrics in sorted(by_step.items()):
        mlflow.log_metrics(metrics, step=step) # log_metrics per step, not per (key, step) pair, asnc :synchronous=False
 
def _log_scalar_metrics(results: dict) -> None:
    metrics = {}
    for key, value in results.items():
        fval = _safe_float(value)
        if fval is not None:
            metrics[key] = fval
    if metrics:
        mlflow.log_metrics(metrics)  #sync/asnc :synchronous=False
 
 
# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------
 
 
def log_results(
    run_name: str,
    grid_params: dict,
    tuned_params: dict,
    cv_results: dict,
    estimators: list[BaseEstimator],
) -> None:
    n_folds = len(estimators)
    fold_metric_keys = [k for k in cv_results if k.startswith("test_")]
    extra_keys = [k for k in ("fit_time", "score_time") if k in cv_results]
 
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("grid_point", True)
 
        # Grid-Parameter
        mlflow.log_params({f"grid_{k}": str(v) for k, v in grid_params.items()})
 
        # Beste Optuna-Parameter (Objekte → str)
        if tuned_params:
            serialisable = {
                k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                for k, v in tuned_params.items()
            }
            mlflow.log_params({f"tuned_{k}": v for k, v in serialisable.items()})
 
        # Nested Runs pro Fold
        for i, estimator in enumerate(estimators):
            fold_result: dict = {}
            for k in fold_metric_keys + extra_keys:
                arr = cv_results.get(k)
                if arr is not None:
                    fold_result[k] = float(arr[i])
 
            # Skalare aus y_scaler-Einträgen (skalare Werte, keine Arrays)
            for k, v in cv_results.items():
                if k not in fold_metric_keys + extra_keys and np.isscalar(v):
                    fold_result[k] = float(v)
 
            with mlflow.start_run(
                run_name=f"{run_name}.fold-{i + 1}-of-{n_folds}", nested=True
            ):
                mlflow.set_tag("fold", True)
                mlflow.set_tag("fold_index", i + 1)
                _log_estimator_run(estimator)
                _log_scalar_metrics(fold_result)
                #mlflow.flush_async_logging() #async logging flush
 
        # Aggregierte Metriken im Parent-Run
        aggregated: dict[str, float] = {}
        for k, v in cv_results.items():
            arr = np.asarray(v, dtype=float) if not np.isscalar(v) else np.array([float(v)])
            aggregated[f"{k}_mean"] = float(np.mean(arr))
            if arr.size > 1:
                aggregated[f"{k}_std"] = float(np.std(arr))
 
        mlflow.log_metrics(aggregated)
 
        # Übersicht ausgeben
        for k in fold_metric_keys:
            arr = np.asarray(cv_results[k], dtype=float)
            print(
                f"[mlflow] {k}: mean={np.mean(arr):.4f} ± {np.std(arr):.4f}"
            )

def log_results_multi_seed(
    run_name: str,
    grid_params: dict,
    tuned_params: dict,
    seed_results: list[tuple[list, dict]],   # [(estimators, cv_results), ...]
    random_states: list,
    walltime_metrics: Optional[dict],
) -> None:
    n_seeds = len(seed_results)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("grid_point", True)
        mlflow.log_params({f"grid_{k}": str(v) for k, v in grid_params.items()})
        mlflow.log_param("n_seeds", n_seeds)

        if tuned_params:
            serialisable = {
                k: str(v) if not isinstance(v, (int, float, str, bool)) else v
                for k, v in tuned_params.items()
            }
            mlflow.log_params({f"tuned_{k}": v for k, v in serialisable.items()})

        seed_averages: list[dict] = []

        for i, (estimators, cv_results) in enumerate(seed_results):
            rs = int(random_states[i])
            n_folds = len(estimators)
            fold_metric_keys = [k for k in cv_results if k.startswith("test_")]
            extra_keys = [k for k in ("fit_time", "score_time") if k in cv_results]

            with mlflow.start_run(run_name=f"{run_name}.seed-{i}", nested=True):
                mlflow.set_tag("seed_run", True)
                mlflow.set_tag("seed_index", i)
                mlflow.set_tag("random_state", rs)

                # Fold-Runs
                for j, estimator in enumerate(estimators):
                    fold_result = {}
                    for k in fold_metric_keys + extra_keys:
                        arr = cv_results.get(k)
                        if arr is not None:
                            fold_result[k] = float(arr[j])
                    for k, v in cv_results.items():
                        if k not in fold_metric_keys + extra_keys and np.isscalar(v):
                            fold_result[k] = float(v)

                    with mlflow.start_run(
                        run_name=f"{run_name}.seed-{i}.fold-{j+1}-of-{n_folds}",
                        nested=True,
                    ):
                        mlflow.set_tag("fold", True)
                        mlflow.set_tag("fold_index", j + 1)
                        _log_estimator_run(estimator)
                        _log_scalar_metrics(fold_result)

                # Gemittelte CV-Scores für diesen Seed + std
                seed_metrics = {}
                for k, v in cv_results.items():
                    arr = np.asarray(v, dtype=float)
                    seed_metrics[f"{k}_mean"] = float(np.mean(arr))
                    if arr.size > 1:
                        seed_metrics[f"{k}_std"] = float(np.std(arr))
                _log_scalar_metrics(seed_metrics)
                seed_averages.append(seed_metrics)

        # Über alle Seeds mitteln → Parent Run
        all_keys = set().union(*seed_averages)
        aggregated: dict[str, float] = {}
        for k in all_keys:
            vals = [d[k] for d in seed_averages if k in d]
            aggregated[f"{k}_mean"] = float(np.mean(vals))
            if len(vals) > 1:
                aggregated[f"{k}_std"] = float(np.std(vals))

        mlflow.log_metrics(aggregated)

        if walltime_metrics:
            mlflow.log_metrics({k: float(v) for k, v in walltime_metrics.items()})

        for k in [k for k in all_keys if k.startswith("test_")]:
            vals = [d[k] for d in seed_averages if k in d]
            print(f"[mlflow] {k}: mean={np.mean(vals):.4f} ± {np.std(vals):.4f}")
        if walltime_metrics:
            for k, v in walltime_metrics.items():
                print(f"[mlflow] {k}: {v:.2f} s")


