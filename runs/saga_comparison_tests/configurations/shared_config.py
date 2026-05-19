from sklearn.utils import Bunch
import numpy as np
from sklearn.linear_model import Ridge
from suprb import SupRB, rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import mutation, es, origin


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets

    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def get_optimizer(name: str) -> type:
    return {
        "ga": ga.GeneticAlgorithm,
    }[name]()


random_state = 42

estimator = SupRB(
    rule_discovery=es.ES1xLambda(
        init=rule.initialization.MeanInit(
            fitness=rule.fitness.VolumeWu(), model=Ridge(alpha=0.01, random_state=random_state)
        ),
        mutation=mutation.HalfnormIncrease(),
    ),
    n_iter=32,
    n_rules=4,
    verbose=10,
    logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
)

# Default values to be used for all tunings
shared_tuning_params = dict(
    estimator=estimator,
    random_state=random_state,
    cv=4,
    n_jobs_cv=4,
    n_jobs=4,
    n_calls=10000,
    timeout=72 * 60 * 60,  # 72 hours
    verbose=10,
)
