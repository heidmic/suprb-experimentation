import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import Bunch
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
        'ga': ga.GeneticAlgorithm,
    }[name]()


random_state = 42

global_params = Bunch(**{
    'solution_composition__n_iter': 32,
    'solution_composition__population_size': 32,
})

dataset_params = {
    'combined_cycle_power_plant': {
        'rule_generation__init__fitness__alpha': 0.05,
        'rule_generation__mutation': mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 0.58,
        'rule_generation__delay': 84,
    },
    'concrete_strength': {
        'rule_generation__init__fitness__alpha': 0.07,
        'rule_generation__mutation': mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 2.62,
        'rule_generation__delay': 124,
    },
    'airfoil_self_noise': {
        'rule_generation__init__fitness__alpha': 0.05,
        'rule_generation__mutation': mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 2.01,
        'rule_generation__delay': 146,
    },
    'energy_cool': {
        'rule_generation__init__fitness__alpha': 0.05,
        'rule_generation__mutation': mutation.HalfnormIncrease(),
        'rule_generation__mutation__sigma': 1.22,
        'rule_generation__delay': 69,
    },
}

optimizer_params = {
    'combined_cycle_power_plant': {
        'ga': {
            'solution_composition__selection': ga.selection.Tournament(),
            'solution_composition__selection__k': 6,
            'solution_composition__crossover': ga.crossover.NPoint(),
            'solution_composition__crossover__n': 5,
            'solution_composition__mutation__mutation_rate': 0.026,
            'solution_composition__elitist_ratio': 0.17,  # 5
        },
    },
    'airfoil_self_noise': {
        'ga': {
            'solution_composition__selection': ga.selection.LinearRank(),
            'solution_composition__crossover': ga.crossover.NPoint(),
            'solution_composition__crossover__n': 3,
            'solution_composition__mutation__mutation_rate': 0.001,
            'solution_composition__elitist_ratio': 0.17,  # 5
        },
    },
    'concrete_strength': {
        'ga': {
            'solution_composition__selection': ga.selection.Tournament(),
            'solution_composition__selection__k': 5,
            'solution_composition__crossover': ga.crossover.Uniform(),
            'solution_composition__mutation__mutation_rate': 0.024,
            'solution_composition__elitist_ratio': 0.16,  # 5
        },
    },

}

estimator = SupRB(
    rule_generation=es.ES1xLambda(
        operator='&',
        n_iter=1,
        init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                          model=Ridge(alpha=0.01, random_state=random_state)),
        mutation=mutation.HalfnormIncrease(),
        origin_generation=origin.SquaredError(),
    ),
    solution_composition=ga.GeneticAlgorithm(),
    n_iter=12,
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
    n_calls=64,
    timeout=90 * 60 * 60,  # 90 hours
    verbose=10
)
