import numpy as np
from sklearn.linear_model import Ridge
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
    'online_news': {
        'n_iter': 32,
        'n_rules': 6
    },
    'protein_structure': {
        'n_iter': 32,
        'n_rules': 6
    },
    'superconductivity': {
        'n_iter': 32,
        'n_rules': 6
    }
}

# TODO Set optimal values, once tuning is finished

dataset_params = {
    'combined_cycle_power_plant': {

    },
    'concrete_strength': {

    },
    'airfoil_self_noise': {

    },
    'online_news': {

    },
    'protein_structure': {

    },
    'superconductivity': {

    }
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
    'online_news': {
        'ga': {
            'solution_composition__selection': ga.selection.Tournament(),
            'solution_composition__selection__k': 5,
            'solution_composition__crossover': ga.crossover.Uniform(),
            'solution_composition__mutation__mutation_rate': 0.024,
            'solution_composition__elitist_ratio': 0.16,  # 5
        },
    },
    'protein_structure': {
        'ga': {
            'solution_composition__selection': ga.selection.Tournament(),
            'solution_composition__selection__k': 5,
            'solution_composition__crossover': ga.crossover.Uniform(),
            'solution_composition__mutation__mutation_rate': 0.024,
            'solution_composition__elitist_ratio': 0.16,  # 5
        },
    },
    'superconductivity': {
        'ga': {
            'solution_composition__selection': ga.selection.Tournament(),
            'solution_composition__selection__k': 5,
            'solution_composition__crossover': ga.crossover.Uniform(),
            'solution_composition__mutation__mutation_rate': 0.024,
            'solution_composition__elitist_ratio': 0.16,  # 5
        },
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
