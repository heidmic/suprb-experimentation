import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import Bunch
from suprb import SupRB, rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import mutation, es, origin
from suprb.rule import initialization


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

# Values that will be chosen as default for all combinations
global_params = Bunch(**{
    'solution_composition__n_iter': 32,
    'rule_generation__n_iter': 250,
    'solution_composition__population_size': 32,
    'rule_generation__lmbda': 20,
    'solution_composition__elitist_ratio': 0.17
})

individual_dataset_params = {
    'airfoil_self_noise': {
        'n_iter': 32,
        'n_rules': 4
    },
    'combined_cycle_power_plant': {
        'n_iter': 32,
        'n_rules': 4
    },
    'concrete_strength': {
        'n_iter': 32,
        'n_rules': 4
    },
    'protein_structure': {
        'n_iter': 36,
        'n_rules': 4
    },
    'parkinson_total': {
        'n_iter': 36,
        'n_rules': 4
    }
}

# TODO Set optimal values, once tuning is finished

dataset_params = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma_center': 0.2817,
        'rule_generation__mutation__sigma_spread': 2.8721,
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma_center': 1.4156,
        'rule_generation__init__sigma_spread': 2.2777,
        'rule_generation__init__fitness__alpha': 0.0313,
        'rule_generation__delay': 91,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 10,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 5,
        'solution_composition__mutation__mutation_rate': 0.0205
    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma_center': 0.0002,
        'rule_generation__mutation__sigma_spread': 0.7830,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0411,
        'rule_generation__delay': 30,
        'solution_composition__selection': getattr(ga.selection, 'Random')(),
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 5,
        'solution_composition__mutation__mutation_rate': 0.0079
    },
    'concrete_strength': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma_center': 0.0004,
        'rule_generation__mutation__sigma_spread': 2.0391,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0228,
        'rule_generation__delay': 6,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.010
    },
    'protein_structure': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma_center': 0.0073,
        'rule_generation__mutation__sigma_spread': 2.5819,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0120,
        'rule_generation__delay': 86,
        'solution_composition__selection': getattr(ga.selection, 'Random')(),
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 6,
        'solution_composition__mutation__mutation_rate': 0.0086
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma_center': 0.0008,
        'rule_generation__mutation__sigma_spread': 0.3103,
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma_center': 0.1263,
        'rule_generation__init__sigma_spread': 2.2086,
        'rule_generation__init__fitness__alpha': 0.0650,
        'rule_generation__delay': 25,
        'solution_composition__selection': getattr(ga.selection, 'Random')(),
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 7,
        'solution_composition__mutation__mutation_rate': 0.0283
    }
}

estimator = SupRB(
    rule_generation=es.ES1xLambda(
        operator='&',
        lmbda=20,
        init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                          model=Ridge(alpha=0.01, random_state=random_state)),
        mutation=mutation.HalfnormIncrease(),
        origin_generation=origin.SquaredError(),
    ),
    solution_composition=ga.GeneticAlgorithm(),
    verbose=10,
    logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]),
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
    verbose=10
)