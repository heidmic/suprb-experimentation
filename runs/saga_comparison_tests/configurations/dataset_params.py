import numpy as np
from suprb.optimizer.solution import ga, saga1, saga2, saga3
from suprb.optimizer.rule import mutation
from suprb.rule import initialization


params_ga = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation__sigma': 0.6859009365594104,
        'rule_generation__init__fitness__alpha': 0.02333434010643204,
        'solution_composition': ga.GeneticAlgorithm(),
        'solution_composition__selection': getattr(ga.selection, 'Tournament')(),
        'solution_composition__selection__k': 9,
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(),
        'solution_composition__crossover__crossover_rate': 0.914091860535504,
        'solution_composition__mutation__mutation_rate': 0.02575478849665418
    },
    'airfoil_self_noise': {
        'rule_generation__mutation__sigma': 2.105582184769341, 
        'rule_generation__init__fitness__alpha': 0.02199191731493414, 
        'solution_composition': ga.GeneticAlgorithm(), 
        'solution_composition__selection': getattr(ga.selection, 'RouletteWheel')(), 
        'solution_composition__mutation__mutation_rate': 0.02148100441946754, 
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(), 
        'solution_composition__crossover__crossover_rate': 0.9830664952156787, 
        'solution_composition__crossover__n': 9
    },
    'concrete_strength': {
        'rule_generation__mutation__sigma': 2.6208139828922463, 
        'rule_generation__init__fitness__alpha': 0.05454544362563632, 
        'solution_composition': ga.GeneticAlgorithm(), 
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(), 
        'solution_composition__mutation__mutation_rate': 0.06027139887171315, 
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(), 
        'solution_composition__crossover__crossover_rate': 0.807403554066105, 
        'solution_composition__crossover__n': 2
    },
    'protein_structure': {
        'rule_generation__mutation__sigma': 2.629758097716877, 
        'rule_generation__init__fitness__alpha': 0.015217998535377692, 
        'solution_composition': ga.GeneticAlgorithm(), 
        'solution_composition__selection': getattr(ga.selection, 'LinearRank')(), 
        'solution_composition__mutation__mutation_rate': 0.03500455204655202, 
        'solution_composition__crossover': getattr(ga.crossover, 'NPoint')(),
        'solution_composition__crossover__crossover_rate': 0.8183035444482907, 
        'solution_composition__crossover__n': 2
    },
    'parkinson_total': {
        'rule_generation__mutation__sigma': 4.06230815699017, 
        'rule_generation__init__fitness__alpha': 0.010116533521591767, 
        'solution_composition': ga.GeneticAlgorithm(), 
        'solution_composition__selection': getattr(ga.selection, 'Random')(),  
        'solution_composition__mutation__mutation_rate': 0.019974012838769737, 
        'solution_composition__crossover': getattr(ga.crossover, 'Uniform')(), 
        'solution_composition__crossover__crossover_rate': 0.7222524054023435
    }
}

params_saga1 = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation__sigma': 0.4276687513599815, 
        'rule_generation__init__fitness__alpha': 0.012714825120382297, 
        'solution_composition': saga1.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 26, 
        'solution_composition__selection': getattr(saga1.selection, 'RouletteWheel')(), 
        'solution_composition__crossover': getattr(saga1.crossover, 'Uniform')()
    },
    'airfoil_self_noise': {
        'rule_generation__mutation__sigma': 2.06919494558429, 
        'rule_generation__init__fitness__alpha': 0.01994191931892904, 
        'solution_composition': saga1.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 98, 
        'solution_composition__selection': getattr(saga1.selection, 'Random')(), 
        'solution_composition__crossover': getattr(saga1.crossover, 'NPoint')(), 
        'solution_composition__crossover__n': 4
    },
    'concrete_strength': {
        'rule_generation__mutation__sigma': 2.551316064860687, 
        'rule_generation__init__fitness__alpha': 0.06496686771281035, 
        'solution_composition': saga1.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 92, 
        'solution_composition__selection': getattr(saga1.selection, 'Random')(), 
        'solution_composition__crossover': getattr(saga1.crossover, 'NPoint')(), 
        'solution_composition__crossover__n': 10
    },
    'protein_structure': {
        'rule_generation__mutation__sigma': 2.854811111764082, 
        'rule_generation__init__fitness__alpha': 0.01676428321108141, 
        'solution_composition': saga1.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 68, 
        'solution_composition__selection': getattr(saga1.selection, 'Tournament')(), 
        'solution_composition__selection__k': 10, 
        'solution_composition__crossover': getattr(saga1.crossover, 'NPoint')(), 
        'solution_composition__crossover__n': 4
    },
    'parkinson_total': {
        'rule_generation__mutation__sigma': 4.213041255163031, 
        'rule_generation__init__fitness__alpha': 0.010265507041105345, 
        'solution_composition': saga1.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 76, 
        'solution_composition__selection': getattr(saga1.selection, 'Tournament')(), 
        'solution_composition__selection__k': 5, 
        'solution_composition__crossover': getattr(saga1.crossover, 'NPoint')(), 
        'solution_composition__crossover__n': 6
    }
}

params_saga2 = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation__sigma': 0.7815096036884479, 
        'rule_generation__init__fitness__alpha': 0.029147681451952713, 
        'solution_composition': saga2.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 32, 
        'solution_composition__selection': getattr(saga2.selection, 'Tournament')(), 
        'solution_composition__selection__k': 8, 
        'solution_composition__crossover': getattr(saga2.crossover, 'Uniform')()
    },
    'airfoil_self_noise': {
        'rule_generation__mutation__sigma': 2.131483810985764, 
        'rule_generation__init__fitness__alpha': 0.016485254492219836, 
        'solution_composition': saga2.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 124, 
        'solution_composition__selection': getattr(saga2.selection, 'LinearRank')(), 
        'solution_composition__crossover': getattr(saga2.crossover, 'Uniform')()
    },
    'concrete_strength': {
        'rule_generation__mutation__sigma': 2.4787655878228, 
        'rule_generation__init__fitness__alpha': 0.02892446714951632, 
        'solution_composition': saga2.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 54, 
        'solution_composition__selection': getattr(saga2.selection, 'Random')(), 
        'solution_composition__crossover': getattr(saga2.crossover, 'NPoint')(), 
        'solution_composition__crossover__n': 1
    },
    'protein_structure': {
        'rule_generation__mutation__sigma': 1.9513238278283638, 
        'rule_generation__init__fitness__alpha': 0.010428435367366349, 
        'solution_composition': saga2.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 124, 
        'solution_composition__selection': getattr(saga2.selection, 'LinearRank')(), 
        'solution_composition__crossover': getattr(saga2.crossover, 'NPoint')(), 
        'solution_composition__crossover__n': 10
    },
    'parkinson_total': {
        'rule_generation__mutation__sigma': 3.8987614190145994, 
        'rule_generation__init__fitness__alpha': 0.01028134204963452, 
        'solution_composition': saga2.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 52, 
        'solution_composition__selection': getattr(saga2.selection, 'Random')(), 
        'solution_composition__crossover': getattr(saga2.crossover, 'Uniform')()
    }
}

params_saga3 = {
    'combined_cycle_power_plant': {
        'rule_generation__mutation__sigma': 1.8012460475029914, 
        'rule_generation__init__fitness__alpha': 0.017014904014171738, 
        'solution_composition': saga3.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 122, 
        'solution_composition__selection': getattr(saga3.selection, 'Random')()
    },
    'airfoil_self_noise': 	{
        'rule_generation__mutation__sigma': 2.234427912602256, 
        'rule_generation__init__fitness__alpha': 0.01585503400362051, 
        'solution_composition': saga3.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 106, 
        'solution_composition__selection': getattr(saga3.selection, 'Random')()
    },
    'concrete_strength': {
        'rule_generation__mutation__sigma': 2.638904587594894, 
        'rule_generation__init__fitness__alpha': 0.0566214425179575, 
        'solution_composition': saga3.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 122, 
        'solution_composition__selection': getattr(saga3.selection, 'LinearRank')()
    },
    'protein_structure': {
        'rule_generation__mutation__sigma': 2.466628786586988, 
        'rule_generation__init__fitness__alpha': 0.018040735824937, 
        'solution_composition': saga3.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 62, 
        'solution_composition__selection': getattr(saga3.selection, 'Random')()
    },
    'parkinson_total': {
        'rule_generation__mutation__sigma': 3.848300857584986, 
        'rule_generation__init__fitness__alpha': 0.010527440178672106, 
        'solution_composition': saga3.SelfAdaptingGeneticAlgorithm(), 
        'solution_composition__n_iter': 114, 
        'solution_composition__selection': getattr(saga3.selection, 'LinearRank')()}
}
