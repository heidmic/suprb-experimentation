import math

import mlflow
import numpy as np
from optuna import Trial
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from suprb2 import rule, SupRB2
from suprb2.logging.combination import CombinedLogger
from suprb2.logging.default import DefaultLogger
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.individual import ga
from suprb2.optimizer.rule import es
from suprb2opt.individual import gwo, aco, pso, abc, rs

from experiments import Experiment
from experiments.evaluation import CrossValidateTest
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space, individual_optimizer_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y
from problems.functions import load_test_function
from problems.functions.fixed import higdon_gramacy_lee

if __name__ == '__main__':
    random_state = 42

    X, y = load_test_function(higdon_gramacy_lee, noise=0.1, random_state=random_state)
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    global_params = Bunch(**{
        'individual_optimizer__n_iter': 32,
        'individual_optimizer__population_size': 128,
    })

    estimator = SupRB2(
        rule_generation=es.ES1xLambda(),
        individual_optimizer=ga.GeneticAlgorithm(),
        n_iter=16,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )

    shared_tuning_params = dict(
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=64,
        verbose=10
    )

    default_tuner = OptunaTuner(**shared_tuning_params)
    fitness_tuner = OptunaTuner(scoring='fitness', **shared_tuning_params)

    experiment = Experiment(name='Individual Optimizer', params=global_params, n_jobs=4, verbose=10)


    @param_space('rule_generation')
    def rule_generation_space(trial: Trial, params: Bunch):
        sigma_space = [0, math.sqrt(X.shape[1])]

        params.mutation = trial.suggest_categorical('mutation', ['HalfnormIncrease', 'Normal'])
        params.mutation = getattr(es.mutation, params.mutation)()
        params.mutation.sigma = trial.suggest_float('mutation__sigma', *sigma_space)

        if isinstance(params.mutation, es.mutation.HalfnormIncrease):
            params.init = rule.initialization.MeanInit()
        else:
            params.init = rule.initialization.HalfnormInit()

            params.init.sigma = trial.suggest_float('init__sigma', *sigma_space)


    experiment.with_tuning(rule_generation_space, tuner=default_tuner)

    # GA
    ga_experiment = experiment.with_params({'individual_optimizer': ga.GeneticAlgorithm()})
    ga_experiment.name = 'GA'


    @individual_optimizer_space
    def ga_space(trial: Trial, params: Bunch):
        params.selection = trial.suggest_categorical('selection',
                                                     ['RouletteWheel', 'Tournament', 'LinearRank', 'Random'])
        params.selection = getattr(ga.selection, params.selection)()

        params.crossover = trial.suggest_categorical('crossover', ['NPoint', 'Uniform'])
        params.crossover = getattr(ga.crossover, params.crossover)()

        params.mutation__mutation_rate = trial.suggest_float('mutation_rate', 0, 0.1)


    ga_experiment.with_tuning(ga_space, tuner=fitness_tuner)

    # GWO
    gwo_experiment = experiment.with_params({'individual_optimizer': gwo.GreyWolfOptimizer()})
    gwo_experiment.name = 'GWO'


    @individual_optimizer_space
    def gwo_space(trial: Trial, params: Bunch):
        params.position = trial.suggest_categorical('position', ['Sigmoid', 'Crossover'])
        params.position = getattr(gwo.position, params.position)()


    gwo_experiment.with_tuning(gwo_space, tuner=fitness_tuner)

    # ACO
    aco_experiment = experiment.with_params({'individual_optimizer': aco.AntColonyOptimization()})
    aco_experiment.name = 'ACO'


    @individual_optimizer_space
    def aco_space(trial: Trial, params: Bunch):
        params.builder = trial.suggest_categorical('builder', ['Binary', 'Complete'])
        params.builder = getattr(aco.builder, params.builder)()
        params.builder.alpha = trial.suggest_float('alpha', 0.5, 5)
        params.builder.beta = trial.suggest_float('beta', 0.5, 5)

        params.evaporation_rate = trial.suggest_float('evaporation_rate', 0, 1)
        params.selection__n = trial.suggest_int('selection__n', 1,
                                                global_params.individual_optimizer__population_size // 2)


    aco_experiment.with_tuning(aco_space, tuner=fitness_tuner)

    # PSO
    pso_experiment = experiment.with_params({'individual_optimizer': pso.ParticleSwarmOptimization()})
    pso_experiment.name = 'PSO'


    @individual_optimizer_space
    def pso_space(trial: Trial, params: Bunch):
        params.movement = trial.suggest_categorical('movement', ['Sigmoid', 'SigmoidQuantum', 'BinaryQuantum'])
        params.movement = getattr(pso.movement, params.movement)()

        params.a_min = trial.suggest_float('a_min', 0, 3)
        params.a_max = trial.suggest_float('a_max', params.a_min, 3)

        if isinstance(params.movement, pso.movement.Sigmoid):
            params.movement.b = trial.suggest_float('b', 0, 3)
            params.movement.c = trial.suggest_float('c', 0, 3)
        elif isinstance(params.movement, pso.movement.BinaryQuantum):
            params.movement.p_learning = trial.suggest_float('p_learning', 0.01, 1)
            params.movement.n_attractors = trial.suggest_int('n_attractors', 1,
                                                             global_params.individual_optimizer__population_size // 2)


    pso_experiment.with_tuning(pso_space, tuner=fitness_tuner)

    # ABC
    abc_experiment = experiment.with_params({'individual_optimizer': abc.ArtificialBeeColonyAlgorithm()})
    abc_experiment.name = 'ABCA'


    @individual_optimizer_space
    def abc_space(trial: Trial, params: Bunch):
        params.food = trial.suggest_categorical('food', ['Sigmoid', 'Bitwise', 'DimensionFlips'])
        params.food = getattr(abc.food, params.food)()

        params.trials_limit = trial.suggest_int('trials_limit', 1, global_params.individual_optimizer__n_iter)

        if isinstance(params.food, abc.food.DimensionFlips):
            params.food.flip_rate = trial.suggest_float('b', 0.01, 1)


    abc_experiment.with_tuning(abc_space, tuner=fitness_tuner)

    # RS
    rs_experiment = experiment.with_params({'individual_optimizer': rs.RandomSearch()})
    rs_experiment.name = 'RS'

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    # Evaluation
    evaluation = CrossValidateTest(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=8, n_jobs=8)

    mlflow.set_experiment("Higdon & Gramacy & Lee")
    log_experiment(experiment)
