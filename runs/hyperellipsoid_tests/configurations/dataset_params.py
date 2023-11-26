import numpy as np
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import mutation
from suprb.rule import initialization


params_ellipsoid = {
    'combined_cycle_power_plant': {

    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': np.array([2.1005, 1.4685]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array([0.55455251, 1.31679535]),
        'rule_generation__init__fitness__alpha': 0.037735774746396916,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 45,
        'solution_composition__selection': getattr(ga.selection, 'RouletteWheel')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.007680268561205435
    },
    'concrete_strength': {

    },
    'energy_cool': {

    },
    'parkinson_total': {

    }
}

params_general_ellipsoid = {
    'combined_cycle_power_plant': {

    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma': np.array([2.2909, 1.2053]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array(([0.55764735, 2.13198119])),
        'rule_generation__init__fitness__alpha': 0.048939179265470395,
        'rule_generation__operator': '&',
        'rule_generation__delay': 14,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 8,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.047255634743850396
    },
    'concrete_strength': {

    },
    'energy_cool': {

    },
    'parkinson_total': {

    }
}

params_ordered_bound = {
    'combined_cycle_power_plant': {

    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma': 1.4984,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0362,
        'rule_generation__operator': '+',
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 3,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 3,
        'solution_composition__mutation__mutation_rate': 0.0128
    },
    'concrete_strength': {

    },
    'energy_cool': {

    },
    'parkinson_total': {

    }
}