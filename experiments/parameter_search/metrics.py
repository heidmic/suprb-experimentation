from suprb import SupRB
from sklearn.base import BaseEstimator
from suprb.logging.combination import CombinedLogger
from suprb.logging.multi_objective import MOLogger
from typing import Optional


__all__ = ["fitness", "train_hypervolume"]


def _get_default_logger(estimator: BaseEstimator) -> Optional[MOLogger]:
    if isinstance(estimator, SupRB):
        logger = estimator.logger_
        if isinstance(logger, MOLogger):
            return logger
        elif isinstance(logger, CombinedLogger):
            for name, sublogger in logger.loggers_:
                if isinstance(sublogger, MOLogger):
                    return sublogger


def fitness(model: SupRB):
    return model.elitist_.fitness_


def train_hypervolume(model: SupRB):
    # Todo: This cant be the right way to do this
    logger = _get_default_logger(model)
    if not logger:
        raise AttributeError("Training Hypervolume was chosen as metric, but no MOLogger was found.")
    hv_dict = logger.metrics_["train_hypervolume"]
    return hv_dict[list(hv_dict.keys())[-1]] if len(hv_dict) > 0 else 0
