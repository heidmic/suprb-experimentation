from suprb.optimizer.solution import ga
from suprb.optimizer.rule import mutation
from suprb.rule import initialization


params_obr = {
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

params_ubr = {
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

params_csr = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': [0.0810, 2.5713],
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': [2.6555, 0.5714],
        'rule_generation__init__fitness__alpha': 0.0968,
        'rule_generation__operator': ',',
        'rule_generation__n_iter': 12,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 10,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0236
    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': [0.0012, 0.9136],
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0356,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 33,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0056
    },
    'concrete_strength': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': [0.0025, 0.0983],
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0327,
        'rule_generation__operator': '&',
        'rule_generation__delay': 23,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 5,
        'solution_composition__mutation__mutation_rate': 0.0146
    },
    'protein_structure': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': [0.0010, 0.1596],
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': [0.0896, 0.3163],
        'rule_generation__init__fitness__alpha': 0.0016,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 30,
        'solution_composition__selection': getattr(ga.selection, 'RouletteWheel')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0122
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': [0.0005, 2.6483],
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0028,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 48,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0085
    }
}

params_mpr = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation': getattr(mutation, 'Normal')(),
        'rule_generation__mutation__sigma': [2.9949, 2.7095],
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0225,
        'rule_generation__operator': ',',
        'rule_generation__n_iter': 46,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 10,
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 2,
        'solution_composition__mutation__mutation_rate': 0.0256
    },
    'airfoil_self_noise': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': [2.8271, 2.3035],
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0056,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 47,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__n': 2,
        'solution_composition__mutation__mutation_rate': 0.0165
    },
    'concrete_strength': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': [2.8565, 2.7411],
        'rule_generation__init': getattr(initialization, 'NormalInit')(),
        'rule_generation__init__sigma': [0.0501, 1.4967],
        'rule_generation__init__fitness__alpha': 0.0855,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 42,
        'solution_composition__selection': getattr(ga.selection, 'RouletteWheel')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0094
    },
    'protein_structure': {
        'rule_generation__mutation': getattr(mutation, 'HalfnormIncrease')(),
        'rule_generation__mutation__sigma': [0.0567, 0.1027],
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0029,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 48,
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(),
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0143
    },
    'parkinson_total': {
        'rule_generation__mutation': getattr(mutation, 'UniformIncrease')(),
        'rule_generation__mutation__sigma': [0.4429, 1.7216],
        'rule_generation__init': getattr(initialization, 'MeanInit')(),
        'rule_generation__init__fitness__alpha': 0.0107,
        'rule_generation__operator': '+',
        'rule_generation__n_iter': 8,
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 6,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__mutation__mutation_rate': 0.0218
    }
}