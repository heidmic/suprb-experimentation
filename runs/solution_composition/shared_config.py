import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import Bunch
from suprb2 import rule, SupRB2, optimizer
from suprb2.logging.combination import CombinedLogger
from suprb2.logging.default import DefaultLogger
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.solution import ga
from suprb2.optimizer.rule import ns, es, origin
# from suprb2opt.solution import aco, gwo, pso, abc, rs


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def get_optimizer(name: str) -> type:
    return {
        'ga': ga.GeneticAlgorithm,
        # 'aco': aco.AntColonyOptimization,
        # 'gwo': gwo.GreyWolfOptimizer,
        # 'pso': pso.ParticleSwarmOptimization,
        # 'abc': abc.ArtificialBeeColonyAlgorithm,
        # 'rs': rs.RandomSearch,
    }[name]()


random_state = 42

global_params = Bunch(**{

})

dataset_params = {
    'combined_cycle_power_plant': {

    },
    'concrete_strength': {

    },
    'airfoil_self_noise': {

    },
    'energy_cool': {

    },
}

optimizer_params = {
    'combined_cycle_power_plant': {
        'ga': {
            'solution_composition__selection': optimizer.solution.ga.selection.Tournament(),
            'solution_composition__selection__k': 6,
            'solution_composition__crossover': optimizer.solution.ga.crossover.NPoint(),
            'solution_composition__crossover__n': 5,
            'solution_composition__mutation__mutation_rate': 0.026,
            'solution_composition__elitist_ratio': 0.17,  # 5
        },
    },
    'airfoil_self_noise': {
        'ga': {
            'solution_composition__selection': optimizer.solution.ga.selection.LinearRank(),
            'solution_composition__crossover': optimizer.solution.ga.crossover.NPoint(),
            'solution_composition__crossover__n': 3,
            'solution_composition__mutation__mutation_rate': 0.001,
            'solution_composition__elitist_ratio': 0.17,  # 5
        },
    },
    'concrete_strength': {
        'ga': {
            'solution_composition__selection': optimizer.solution.ga.selection.Tournament(),
            'solution_composition__selection__k': 5,
            'solution_composition__crossover': optimizer.solution.ga.crossover.Uniform(),
            'solution_composition__mutation__mutation_rate': 0.024,
            'solution_composition__elitist_ratio': 0.16,  # 5
        },
    },
    'energy_cool': {
        'ga': {
            'solution_composition__selection': optimizer.solution.ga.selection.Tournament(),
            'solution_composition__selection__k': 9,
            'solution_composition__crossover': optimizer.solution.ga.crossover.Uniform(),
            'solution_composition__mutation__mutation_rate': 0.014,
            'solution_composition__elitist_ratio': 0.19,  # 6
        },
    },
}

estimator = SupRB2(
    rule_generation=ns.NoveltySearch(

    ),
    solution_composition=optimizer.solution.ga.GeneticAlgorithm(),
    n_iter=8,
    n_rules=7,
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
