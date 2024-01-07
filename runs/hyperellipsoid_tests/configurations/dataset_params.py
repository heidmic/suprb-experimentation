import numpy as np
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import mutation
from suprb.rule import initialization


params_ellipsoid = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': np.array([0.72459908, 2.57885785]),
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__sigma': np.array([0.55455251, 1.31679535]),
        'rule_generation__init__fitness__alpha': 0.004808453185623891,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 7,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 10,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.01501930889255245
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
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': np.array([0.2613997 , 1.82700701]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array([[0.23751907, 0.95650243]]),
        'rule_generation__init__fitness__alpha': 0.006140267145589394,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 35,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 8,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.011367081726664232
    },
    'energy_cool': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': np.array([1.14973994, 2.22384904]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array([0.31275299, 2.81872968]),
        'rule_generation__init__fitness__alpha': 0.007925349370798754,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 29,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__mutation__mutation_rate': 0.028109358913381316
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': np.array([1.36870619, 1.97921297]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array([0.31378825, 2.71164097]),
        'rule_generation__init__fitness__alpha': 0.005381288815141153,
        'rule_generation__operator': '&',
        'rule_generation__n_iter': 24,
        'solution_composition__selection': getattr(ga.selection, 'RouletteWheel')(),
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 5,
        'solution_composition__mutation__mutation_rate': 0.0082104122800693
    }
}

params_general_ellipsoid = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma': np.array([0.44706173, 2.60740631]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array(([0.2254711 , 2.60110633])),
        'rule_generation__init__fitness__alpha': 0.011559580856769879,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 49,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 8,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 5,
        'solution_composition__mutation__mutation_rate': 0.022067006825602316
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
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': np.array([1.10526289, 2.84168068]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array(([0.05988013, 1.8472693])),
        'rule_generation__init__fitness__alpha': 0.005206506556975432,
        'rule_generation__operator': '&',
        'rule_generation__delay': 13,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__selection__k': 8,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 7,
        'solution_composition__mutation__mutation_rate': 0.016129675573852753
    },
    'energy_cool': {
        'rule_generation__mutation': getattr(mutation, 'Uniform')(),
        'rule_generation__mutation__sigma': np.array([2.3411243 , 2.78285515]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array(([0.14721717, 2.55518729])),
        'rule_generation__init__fitness__alpha': 0.08026408673196553,
        'rule_generation__operator': '&',
        'rule_generation__delay': 7,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.030716131351965555
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'Uniform')(),
        'rule_generation__mutation__sigma': np.array([2.5209, 1.4523]),
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': np.array(([0.5823035, 2.423201])),
        'rule_generation__init__fitness__alpha': 0.0432909817470395,
        'rule_generation__operator': '&',
        'rule_generation__delay': 17,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 9,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0472234123124
    }
}

params_ordered_bound = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.030588577534017662,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 26,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 9,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 6,
        'solution_composition__mutation__mutation_rate': 0.01614402277000072
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
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.01498008087247176,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 43,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 7,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__mutation__mutation_rate': 0.01343874544458994
    },
    'energy_cool': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0129073649,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 39,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 5,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__mutation__mutation_rate': 0.0123410492
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': 1.4732,
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.05138917566793379,
        'rule_generation__operator': '+',
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 5,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 1,
        'solution_composition__mutation__mutation_rate': 0.017072580982634984
    }
}