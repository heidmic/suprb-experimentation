import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import Bunch
from suprb2 import rule, SupRB2
from suprb2.logging.combination import CombinedLogger
from suprb2.logging.default import DefaultLogger
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.individual import ga
from suprb2.optimizer.rule import es, origin
from suprb2opt.individual import aco, gwo, pso, abc, rs


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def get_optimizer(name: str) -> type:
    return {
        'ga': ga.GeneticAlgorithm,
        'aco': aco.AntColonyOptimization,
        'gwo': gwo.GreyWolfOptimizer,
        'pso': pso.ParticleSwarmOptimization,
        'abc': abc.ArtificialBeeColonyAlgorithm,
        'rs': rs.RandomSearch,
    }[name]()


random_state = 42

global_params = Bunch(**{
    'rule_generation__init__fitness__alpha': 0.1,
    'individual_optimizer__n_iter': 32,
    'individual_optimizer__population_size': 32,
})

dataset_params = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 0.28,
    },
    'concrete_strength': {
        'rule_generation__init__fitness__alpha': 0.3,
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 2.3,
    },
    'airfoil_self_noise': {
        'rule_generation__init__fitness__alpha': 0.3,
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 2.15,
    },
    'energy_cool': {
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 1.17,
    },
    'parkinson_motor': {
        'rule_generation__init__fitness__alpha': 0.3,
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 2.17,
    }
}

estimator = SupRB2(
    rule_generation=es.ES1xLambda(
        operator='&',
        n_iter=10_000,
        delay=20,
        init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu()),
        mutation=es.mutation.HalfnormIncrease(sigma=0.8),
        origin_generation=origin.SquaredError(),
    ),
    individual_optimizer=ga.GeneticAlgorithm(),
    n_iter=32,
    n_rules=4,
    verbose=10,
    logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]),
)

shared_tuning_params = dict(
    estimator=estimator,
    random_state=random_state,
    cv=4,
    n_jobs_cv=4,
    n_jobs=4,
    n_calls=128,
    timeout=90 * 60 * 60,  # 90 hours
    verbose=10
)
