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
    'rule_generation__n_iter': 100,
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
        'rule_generation__mutation__sigma': 2.2561,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0693,
        'rule_generation__delay': 84,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 6,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0184
    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': 2.1400,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0019,
        'rule_generation__delay': 71,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 7,
        'solution_composition__crossover' : getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 5,
        'solution_composition__mutation__mutation_rate': 0.0181
    },
    'concrete_strength': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': 2.6196,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0294,
        'rule_generation__delay': 84,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 6,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0174
    },
    'protein_structure': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma': 2.7480,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0029,
        'rule_generation__delay': 84,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 7,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 8,
        'solution_composition__mutation__mutation_rate': 0.0263
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma': 2.9434,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0018,
        'rule_generation__delay': 2,
        'solution_composition__selection': getattr(ga.selection, 'RouletteWheel')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0037
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
