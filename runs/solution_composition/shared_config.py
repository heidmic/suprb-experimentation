import numpy as np
from sklearn.linear_model import Ridge
from sklearn.utils import Bunch
from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es, origin
from suprbopt.solution import aco, gwo, pso, abc, rs


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets

    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def get_optimizer(name: str) -> type:
    return {
        "ga": ga.GeneticAlgorithm,
        "aco": aco.AntColonyOptimization,
        "gwo": gwo.GreyWolfOptimizer,
        "pso": pso.ParticleSwarmOptimization,
        "abc": abc.ArtificialBeeColonyAlgorithm,
        "rs": rs.RandomSearch,
    }[name]()


random_state = 42

global_params = Bunch(
    **{
        "solution_composition__n_iter": 32,
        "solution_composition__population_size": 32,
    }
)

dataset_params = {
    "combined_cycle_power_plant": {
        "rule_discovery__init__fitness__alpha": 0.05,
        "rule_discovery__mutation": es.mutation.HalfnormIncrease(),
        "rule_discovery__mutation__sigma": 0.58,
        "rule_discovery__delay": 84,
    },
    "concrete_strength": {
        "rule_discovery__init__fitness__alpha": 0.07,
        "rule_discovery__mutation": es.mutation.HalfnormIncrease(),
        "rule_discovery__mutation__sigma": 2.62,
        "rule_discovery__delay": 124,
    },
    "airfoil_self_noise": {
        "rule_discovery__init__fitness__alpha": 0.05,
        "rule_discovery__mutation": es.mutation.HalfnormIncrease(),
        "rule_discovery__mutation__sigma": 2.01,
        "rule_discovery__delay": 146,
    },
    "energy_cool": {
        "rule_discovery__init__fitness__alpha": 0.05,
        "rule_discovery__mutation": es.mutation.HalfnormIncrease(),
        "rule_discovery__mutation__sigma": 1.22,
        "rule_discovery__delay": 69,
    },
}

optimizer_params = {
    "combined_cycle_power_plant": {
        "ga": {
            "solution_composition__selection": ga.selection.Tournament(),
            "solution_composition__selection__k": 6,
            "solution_composition__crossover": ga.crossover.NPoint(),
            "solution_composition__crossover__n": 5,
            "solution_composition__mutation__mutation_rate": 0.026,
            "solution_composition__elitist_ratio": 0.17,  # 5
        },
        "aco": {
            "solution_composition__builder": aco.builder.Complete(alpha=2.33, beta=4.91),
            "solution_composition__evaporation_rate": 0.39,
            "solution_composition__selection__n": 1,
        },
        "gwo": {
            "solution_composition__position": gwo.position.Sigmoid(),
            "solution_composition__n_leaders": 1,
        },
        "pso": {
            "solution_composition__movement": pso.movement.Sigmoid(b=2.23, c=2.20),
            "solution_composition__a_min": 1.04,
            "solution_composition__a_max": 2.54,
        },
        "abc": {
            "solution_composition__food": abc.food.Sigmoid(),
            "solution_composition__trials_limit": 24,
        },
    },
    "airfoil_self_noise": {
        "ga": {
            "solution_composition__selection": ga.selection.LinearRank(),
            "solution_composition__crossover": ga.crossover.NPoint(),
            "solution_composition__crossover__n": 3,
            "solution_composition__mutation__mutation_rate": 0.001,
            "solution_composition__elitist_ratio": 0.17,  # 5
        },
        "aco": {
            "solution_composition__builder": aco.builder.Complete(alpha=1.51, beta=1.18),
            "solution_composition__evaporation_rate": 0.78,
            "solution_composition__selection__n": 3,
        },
        "gwo": {
            # 'solution_composition__position': gwo.position.Crossover(),
            "solution_composition__position": gwo.position.Sigmoid(),
            "solution_composition__n_leaders": 2,
            # 'solution_composition__n_leaders': 11,
        },
        "pso": {
            "solution_composition__movement": pso.movement.Sigmoid(b=2.76, c=2.98),
            "solution_composition__a_min": 2.36,
            "solution_composition__a_max": 2.39,
        },
        "abc": {
            "solution_composition__food": abc.food.Bitwise(),
            "solution_composition__trials_limit": 5,
        },
    },
    "concrete_strength": {
        "ga": {
            "solution_composition__selection": ga.selection.Tournament(),
            "solution_composition__selection__k": 5,
            "solution_composition__crossover": ga.crossover.Uniform(),
            "solution_composition__mutation__mutation_rate": 0.024,
            "solution_composition__elitist_ratio": 0.16,  # 5
        },
        "aco": {
            "solution_composition__builder": aco.builder.Complete(alpha=1.45, beta=1.94),
            "solution_composition__evaporation_rate": 0.66,
            "solution_composition__selection__n": 2,
        },
        "gwo": {
            "solution_composition__position": gwo.position.Sigmoid(),
            "solution_composition__n_leaders": 2,
        },
        "pso": {
            "solution_composition__movement": pso.movement.Sigmoid(b=2.32, c=1.36),
            "solution_composition__a_min": 1.27,
            "solution_composition__a_max": 1.79,
        },
        "abc": {
            "solution_composition__food": abc.food.DimensionFlips(flip_rate=0.13),
            "solution_composition__trials_limit": 17,
        },
    },
    "energy_cool": {
        "ga": {
            "solution_composition__selection": ga.selection.Tournament(),
            "solution_composition__selection__k": 9,
            "solution_composition__crossover": ga.crossover.Uniform(),
            "solution_composition__mutation__mutation_rate": 0.014,
            "solution_composition__elitist_ratio": 0.19,  # 6
        },
        "aco": {
            "solution_composition__builder": aco.builder.Complete(alpha=1.66, beta=1.83),
            "solution_composition__evaporation_rate": 0.67,
            "solution_composition__selection__n": 1,
        },
        "gwo": {
            "solution_composition__position": gwo.position.Sigmoid(),
            "solution_composition__n_leaders": 2,
        },
        "pso": {
            "solution_composition__movement": pso.movement.Sigmoid(b=1.12, c=1.42),
            "solution_composition__a_min": 0.35,
            "solution_composition__a_max": 2.65,
        },
        "abc": {
            "solution_composition__food": abc.food.Sigmoid(),
            "solution_composition__trials_limit": 11,
        },
    },
}

estimator = SupRB(
    rule_discovery=es.ES1xLambda(
        operator="&",
        n_iter=10_000,
        init=rule.initialization.MeanInit(
            fitness=rule.fitness.VolumeWu(), model=Ridge(alpha=0.01, random_state=random_state)
        ),
        mutation=es.mutation.HalfnormIncrease(),
        origin_generation=origin.SquaredError(),
    ),
    solution_composition=ga.GeneticAlgorithm(),
    n_iter=32,
    n_rules=4,
    verbose=10,
    logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
)

shared_tuning_params = dict(
    estimator=estimator,
    random_state=random_state,
    cv=4,
    n_jobs_cv=4,
    n_jobs=4,
    n_calls=128,
    timeout=90 * 60 * 60,  # 90 hours
    verbose=10,
)
