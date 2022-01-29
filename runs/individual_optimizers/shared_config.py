import numpy as np
from sklearn.linear_model import Ridge
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
    'individual_optimizer__n_iter': 32,
    'individual_optimizer__population_size': 32,
})

dataset_params = {
    'combined_cycle_power_plant': {
        'rule_generation__init__fitness__alpha': 0.05,
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 0.58,
        'rule_generation__delay': 84,
    },
    'concrete_strength': {
        'rule_generation__init__fitness__alpha': 0.07,
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 2.62,
        'rule_generation__delay': 124,
    },
    'airfoil_self_noise': {
        'rule_generation__init__fitness__alpha': 0.05,
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 2.01,
        'rule_generation__delay': 146,
    },
    'energy_cool': {
        'rule_generation__init__fitness__alpha': 0.05,
        'rule_generation__mutation': es.mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 1.22,
        'rule_generation__delay': 69,
    },
}

optimizer_params = {
    'combined_cycle_power_plant': {
        'ga': {
            'individual_optimizer__selection': ga.selection.Tournament(),
            'individual_optimizer__selection__k': 5,
            'individual_optimizer__crossover': ga.crossover.Uniform(),
            'individual_optimizer__mutation__mutation_rate': 0.024,
            'individual_optimizer__elitist_ratio': 0.16,  # 5
        },
        'aco': {
            'individual_optimizer__builder': aco.builder.Complete(alpha=2.33, beta=4.91),
            'individual_optimizer__evaporation_rate': 0.39,
            'individual_optimizer__selection__n': 1,
        },
        'gwo': {
            'individual_optimizer__position': gwo.position.Sigmoid(),
            'individual_optimizer__n_leaders': 1,
        },
        'pso': {
            'individual_optimizer__movement': pso.movement.Sigmoid(b=2.23, c=2.20),
            'individual_optimizer__a_min': 1.04,
            'individual_optimizer__a_max': 2.54,
        },
        'abc': {
            'individual_optimizer__food': abc.food.Sigmoid(),
            'individual_optimizer__trials_limit': 24,
        },
    },
    'concrete_strength': {
        'ga': {
            'individual_optimizer__selection': ga.selection.Tournament(),
            'individual_optimizer__selection__k': 5,
            'individual_optimizer__crossover': ga.crossover.Uniform(),
            'individual_optimizer__mutation__mutation_rate': 0.024,
            'individual_optimizer__elitist_ratio': 0.16,  # 5
        },
        'aco': {
            'individual_optimizer__builder': aco.builder.Complete(alpha=1.45, beta=1.94),
            'individual_optimizer__evaporation_rate': 0.66,
            'individual_optimizer__selection__n': 2,
        },
        'gwo': {
            'individual_optimizer__position': gwo.position.Sigmoid(),
            'individual_optimizer__n_leaders': 2,
        },
        'pso': {
            'individual_optimizer__movement': pso.movement.Sigmoid(b=2.32, c=1.36),
            'individual_optimizer__a_min': 1.27,
            'individual_optimizer__a_max': 1.79,
        },
        'abc': {
            'individual_optimizer__food': abc.food.DimensionFlips(flip_rate=0.13),
            'individual_optimizer__trials_limit': 17,
        },

    },
    'airfoil_self_noise': {
        'ga': {
            'individual_optimizer__selection': ga.selection.LinearRank(),
            'individual_optimizer__crossover': ga.crossover.NPoint(),
            'individual_optimizer__crossover__n': 3,
            'individual_optimizer__mutation__mutation_rate': 0.001,
            'individual_optimizer__elitist_ratio': 0.17,  # 5
        },
        'aco':  {'individual_optimizer__builder': aco.builder.Complete(alpha=1.51, beta=1.18),
                 'individual_optimizer__evaporation_rate': 0.78,
                 'individual_optimizer__selection__n': 3,
                 },
        'gwo': {
            # 'individual_optimizer__position': gwo.position.Crossover(),
            'individual_optimizer__position': gwo.position.Sigmoid(),
            'individual_optimizer__n_leaders': 2,
            # 'individual_optimizer__n_leaders': 11,
        },
        'pso': {
            'individual_optimizer__movement': pso.movement.Sigmoid(b=2.76, c=2.98),
            'individual_optimizer__a_min': 2.36,
            'individual_optimizer__a_max': 2.39,
        },
        'abc': {
            'individual_optimizer__food': abc.food.Bitwise(),
            'individual_optimizer__trials_limit': 5,
        },
    },
    'energy_cool': {
        'ga': {
            'individual_optimizer__selection': ga.selection.Tournament(),
            'individual_optimizer__selection__k': 9,
            'individual_optimizer__crossover': ga.crossover.Uniform(),
            'individual_optimizer__mutation__mutation_rate': 0.014,
            'individual_optimizer__elitist_ratio': 0.19,  # 6
        },
        'aco': {
            'individual_optimizer__builder': aco.builder.Complete(alpha=1.66, beta=1.83),
            'individual_optimizer__evaporation_rate': 0.67,
            'individual_optimizer__selection__n': 1
        },
        'gwo': {
            'individual_optimizer__position': gwo.position.Sigmoid(),
            'individual_optimizer__n_leaders': 2,
        },
        'pso': {
            'individual_optimizer__movement': pso.movement.Sigmoid(b=1.12, c=1.42),
            'individual_optimizer__a_min': 0.35,
            'individual_optimizer__a_max': 2.65,
        },
        'abc': {
            'individual_optimizer__food': abc.food.Sigmoid(),
            'individual_optimizer__trials_limit': 11,
        },
    },
}

estimator = SupRB2(
    rule_generation=es.ES1xLambda(
        operator='&',
        n_iter=10_000,
        init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                          model=Ridge(alpha=0.01, random_state=random_state)),
        mutation=es.mutation.HalfnormIncrease(),
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
