import numpy as np
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import mutation
from suprb.rule import initialization


params_ellipsoid = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': 2.8124,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0980,
        'rule_generation__operator': ',',
        'rule_generation__n_iter': 30,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 7,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.04578
    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma': 2.4025,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0116,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 43,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 5,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 7,
        'solution_composition__mutation__mutation_rate': 0.0140
    },
    'concrete_strength': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': 1.6492,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0186,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 50,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 10,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 10,
        'solution_composition__mutation__mutation_rate': 0.0164
    },
    'protein_structure': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': 1.6547,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0001,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 14,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0149
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': 2.2728,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0018,
        'rule_generation__operator': '&',
        'rule_generation__delay': 19,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0093
    }
}

params_general_ellipsoid = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': 2.7917,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0889,
        'rule_generation__operator': ',',
        'rule_generation__n_iter': 23,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 6,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 1,
        'solution_composition__mutation__mutation_rate': 0.0173
    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': 1.4984,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0472,
        'rule_generation__operator': ',',
        'rule_generation__n_iter': 1,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 8,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 9,
        'solution_composition__mutation__mutation_rate': 0.0158
    },
    'concrete_strength': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': 2.1105,
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': 1.46936,
        'rule_generation__init__fitness__alpha': 0.0644,
        'rule_generation__operator': ',',
        'rule_generation__n_iter': 44,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 7,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0157
    },
    'protein_structure': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': 0.3523,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0966,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 5,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 3,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0163
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': 2.3444,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0067,
        'rule_generation__operator': ',',
        'rule_generation__n_iter': 50,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 9,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 3,
        'solution_composition__mutation__mutation_rate': 0.0030
    }
}