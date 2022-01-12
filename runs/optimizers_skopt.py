import math

from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from suprb2 import rule, individual, SupRB2
from suprb2.logging.combination import CombinedLogger
from suprb2.logging.default import DefaultLogger
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.individual import ga
from suprb2.optimizer.rule import es
from suprb2opt.individual import gwo, aco, abc, pso

from experiments import Experiment
from experiments.evaluation import CrossValidateTest
from experiments.parameter_search.skopt import SkoptTuner
from problems import scale_X_y
from problems.datasets import load_concrete_strength

if __name__ == '__main__':
    random_state = 42

    X, y = load_concrete_strength()
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    global_params = {
        'population_optimizer__n_iter': 32,
        'population_optimizer__population_size': 128,
    }

    global_space = {
        'rule_generation__mutation__sigma': Real(0, math.sqrt(X.shape[0])),
    }

    estimator = SupRB2(
        rule_generation=es.ES1xLambda(
            init=rule.initialization.MeanInit(),
            mutation=es.mutation.HalfnormIncrease(),
        ),
        n_iter=4,
        verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )

    tuner = SkoptTuner(
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        n_jobs_cv=4,
        verbose=10
    )

    experiment = Experiment(params=global_params, tuner=tuner)
    experiment.with_tuning(global_space)

    # GA
    ga_experiment = experiment.with_params({'individual_optimizer': ga.GeneticAlgorithm()})
    ga_experiment.with_params({
        'individual_optimizer__selection': [ga.selection.RouletteWheel(), ga.selection.Tournament()],
        'individual_optimizer__crossover': [ga.crossover.NPoint(), ga.crossover.Uniform()],
    })
    ga_experiment.with_tuning({
        'individual_optimizer__mutation__mutation_rate': Real(0, 0.5),
        'individual_optimizer__selection__parent_ratio': Real(0.02, 1),
    }, propagate=True)

    # GWO
    gwo_experiment = experiment.with_params({'individual_optimizer': gwo.GreyWolfOptimizer()})
    gwo_experiment.with_params({
        'population_optimizer__position': [gwo.position.Crossover(), gwo.position.Sigmoid()],
    })

    # ACO
    aco_experiment = experiment.with_params({'individual_optimizer': aco.AntColonyOptimization()})
    aco_experiment.with_params({
        'population_optimizer__builder': [aco.builder.Binary(), aco.builder.Complete()],
    })
    aco_experiment.with_tuning({
        'individual_optimizer__evaporation_rate': Real(0, 0.9),
        'individual_optimizer__selection__n': Integer(1, 50),
        'individual_optimizer__builder__alpha': Real(0.5, 5),
        'individual_optimizer__builder__beta': Real(0.5, 5),
        'individual_optimizer__pheromones_c': Real(0.01, 0.1),
    }, propagate=True)

    # PSO
    pso_experiment = experiment.with_params({'individual_optimizer': pso.ParticleSwarmOptimization()})
    pso_experiment.with_params({'individual_optimizer__movement': pso.movement.Sigmoid()}).with_tuning({
        'individual_optimizer__movement__b': Real(0, 3),
        'individual_optimizer__movement__c': Real(0, 3),
    })
    pso_experiment.with_params({'individual_optimizer__movement': pso.movement.SigmoidQuantum()})
    pso_experiment.with_params({'individual_optimizer__movement': pso.movement.BinaryQuantum()}).with_tuning({
        'individual_optimizer__movement__p_learning': Real(0.01, 1),
        'individual_optimizer__movement__n_attractors': Integer(1, 64),
        'individual_optimizer__movement__mutation_rate': Real(0, 0.5),
    })
    pso_experiment.with_tuning({
        'individual_optimizer__a_min': Real(0, 0.6),
        'individual_optimizer__a_max': Real(0.6, 2),
    }, propagate=True)

    # ABC
    abc_experiment = experiment.with_params({'individual_optimizer': abc.ArtificialBeeColonyAlgorithm()})
    abc_experiment.with_params({'individual_optimizer__food': [abc.food.Sigmoid(), abc.food.Bitwise()]})
    abc_experiment.with_params({'individual_optimizer__food': abc.food.DimensionFlips()}).with_tuning({
        'individual_optimizer__food__flip_rate': Real(0.01, 0.5),
    })
    abc_experiment.with_tuning({
        'individual_optimizer__trials_limit': Integer(1, 128),
    }, propagate=True)

    # Evaluation
    evaluation = CrossValidateTest(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=4, n_jobs=4)
