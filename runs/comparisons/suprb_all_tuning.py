import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga, aco, rs, pso, gwo, abc
from suprb.optimizer.rule import es, origin, mutation, ns, rs
from suprb.solution.initialization import RandomInit
import suprb.solution.mixing_model as mixing_model
import suprb


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-j', '--job_id', type=click.STRING, default='NA')
@click.option('-n', '--study_name', type=click.STRING, default='NoName')
def run(problem: str, job_id: str, study_name: str):
    print(f"Problem is {problem}, with job id {job_id}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_generation=rs.RandomSearch(),
        solution_composition=abc.ArtificialBeeColonyAlgorithm(),
        n_iter=32, n_rules=4, verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()),
                               ('default', DefaultLogger())]),)

    tuning_params = dict(
        study_name=study_name,
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=1000,
        timeout=60*60*24*6,  # 6 days
        scoring='fitness',
        verbose=10
    )

    @param_space()
    def suprb_space(trial: Trial, params: Bunch):
        params.rule_generation = trial.suggest_categorical('rule_generation', ['ES1xLambda', 'NoveltySearch', 'RandomSearch'])  # nopep8

        if params.rule_generation == 'ES1xLambda':
            # ES base
            params.rule_generation = getattr(suprb.optimizer.rule.es, params.rule_generation)()

            params.rule_generation__operator = trial.suggest_categorical('rule_generation__operator', ['&', ',', '+'])

            params.rule_generation__lmbda = trial.suggest_int('rule_generation__lmbda', 10, 30)
            params.rule_generation__n_iter = trial.suggest_int('rule_generation__n_iter', 500, 1500)

            if params.rule_generation__operator == '&':
                params.rule_generation__delay = trial.suggest_int('rule_generation__delay', 20, 50)

        elif params.rule_generation == 'NoveltySearch':
            # NS base
            params.rule_generation = getattr(suprb.optimizer.rule.ns, params.rule_generation)()

            params.rule_generation__lmbda = trial.suggest_int('rule_generation__lmbda', 100, 200)
            params.rule_generation__mu = trial.suggest_int('rule_generation__mu', 10, 20)
            params.rule_generation__n_elitists = trial.suggest_int('rule_generation__n_elitists', 5, 20)
            params.rule_generation__n_iter = trial.suggest_int('rule_generation__n_iter', 5, 15)
            params.rule_generation__roh = trial.suggest_int('rule_generation__roh', 10, 20)

            # NS novelty_calculation
            params.rule_generation__novelty_calculation = trial.suggest_categorical('rule_generation__novelty_calculation', ['NoveltyCalculation', 'ProgressiveMinimalCriteria', 'NoveltyFitnessPareto', 'NoveltyFitnessBiased'])  # nopep8
            params.rule_generation__novelty_calculation = getattr(suprb.optimizer.rule.ns.novelty_calculation, params.rule_generation__novelty_calculation)()  # nopep8

            if isinstance(params.rule_generation__novelty_calculation, suprb.optimizer.rule.ns.novelty_calculation.NoveltyFitnessBiased):  # nopep8
                params.rule_generation__novelty_calculation__novelty_bias = trial.suggest_float('rule_generation__novelty_calculation__novelty_bias', 0.1, 0.9)  # nopep8
            else:
                params.rule_generation__novelty_calculation__k_neighbor = trial.suggest_int('rule_generation__novelty_calculation__k_neighbor', 10, 20)  # nopep8

            params.rule_generation__novelty_calculation__archive = trial.suggest_categorical('rule_generation__novelty_calculation__archive', ['ArchiveNovel', 'ArchiveRandom', 'ArchiveNone'])  # nopep8
            params.rule_generation__novelty_calculation__archive = getattr(suprb.optimizer.rule.ns.archive, params.rule_generation__novelty_calculation__archive)()  # nopep8

            params.rule_generation__novelty_calculation__novelty_search_type = trial.suggest_categorical('rule_generation__novelty_calculation__novelty_search_type', ['NoveltySearchType', 'MinimalCriteria', 'LocalCompetition'])  # nopep8
            params.rule_generation__novelty_calculation__novelty_search_type = getattr(suprb.optimizer.rule.ns.novelty_search_type, params.rule_generation__novelty_calculation__novelty_search_type)()  # nopep8

            if isinstance(params.rule_generation__novelty_calculation__novelty_search_type, suprb.optimizer.rule.ns.novelty_search_type.MinimalCriteria):  # nopep8
                params.rule_generation__novelty_calculation__novelty_search_type__min_examples_matched = trial.suggest_int('rule_generation__novelty_calculation__novelty_search_type__min_examples_matched', 10, 20)  # nopep8
            elif isinstance(params.rule_generation__novelty_calculation__novelty_search_type, suprb.optimizer.rule.ns.novelty_search_type.LocalCompetition):  # nopep8
                params.rule_generation__novelty_calculation__novelty_search_type__max_neighborhood_range = trial.suggest_int('rule_generation__novelty_calculation__novelty_search_type__max_neighborhood_range', 10, 20)  # nopep8
            params.rule_generation__crossover__crossover_rate = trial.suggest_float('rule_generation__crossover__crossover_rate', 0.1, 0.5)  # nopep8
        elif params.rule_generation == 'RandomSearch':
            # RS base
            params.rule_generation = getattr(suprb.optimizer.rule.rs, params.rule_generation)()

            params.rule_generation__n_iter = trial.suggest_int('rule_generation__n_iter', 1, 5)
            params.rule_generation__rules_generated = trial.suggest_int('rule_generation__rules_generated', 5, 10)

        # Acceptance
        params.rule_generation__acceptance = trial.suggest_categorical('rule_generation__acceptance', ['Variance', 'MaxError'])  # nopep8
        params.rule_generation__acceptance = getattr(suprb.optimizer.rule.acceptance, params.rule_generation__acceptance)()  # nopep8

        if isinstance(params.rule_generation__acceptance, suprb.optimizer.rule.acceptance.Variance):
            params.rule_generation__acceptance__beta = trial.suggest_float('rule_generation__acceptance__beta', 1, 3)

        params.rule_generation__constraint = trial.suggest_categorical('rule_generation__constraint', ['MinRange', 'Clip', 'CombinedConstraint'])  # nopep8
        params.rule_generation__constraint = getattr(suprb.optimizer.rule.constraint, params.rule_generation__constraint)()  # nopep8

        if isinstance(params.rule_generation__constraint, suprb.optimizer.rule.constraint.CombinedConstraint):
            params.rule_generation__constraint__clip = suprb.optimizer.rule.constraint.Clip()
            params.rule_generation__constraint__min_range = suprb.optimizer.rule.constraint.MinRange()

        params.rule_generation__init = trial.suggest_categorical('rule_generation__init', ['MeanInit', 'NormalInit', 'HalfnormInit'])  # nopep8
        params.rule_generation__init = getattr(suprb.rule.initialization, params.rule_generation__init)()  # nopep8

        if not isinstance(params.rule_generation__init, suprb.rule.initialization.MeanInit):
            params.rule_generation__init__sigma = trial.suggest_float('rule_generation__init__sigma', 0.01, 0.1)  # nopep8

        params.rule_generation__init__fitness = trial.suggest_categorical('rule_generation__init__fitness', ['PseudoAccuracy', 'VolumeEmary', 'VolumeWu'])  # nopep8
        params.rule_generation__init__fitness = getattr(suprb.rule.fitness, params.rule_generation__init__fitness)()  # nopep8

        if not isinstance(params.rule_generation__init__fitness, suprb.rule.fitness.PseudoAccuracy):
            params.rule_generation__init__fitness__alpha = trial.suggest_float('rule_generation__init__fitness__alpha', 0.5, 1)  # nopep8

        params.rule_generation__init__model = Ridge()

        # Selection
        params.rule_generation__selection = trial.suggest_categorical('rule_generation__selection', ['Fittest', 'RouletteWheel', 'NondominatedSort', 'Random'])  # nopep8
        params.rule_generation__selection = getattr(suprb.optimizer.rule.selection, params.rule_generation__selection)()  # nopep8

        if isinstance(params.rule_generation, suprb.optimizer.rule.es.ES1xLambda) or \
           isinstance(params.rule_generation, suprb.optimizer.rule.ns.NoveltySearch):
            # Mutation
            if params.rule_generation__operator == ',':
                params.rule_generation__mutation = trial.suggest_categorical('rule_generation__mutation__mutation', ['Normal', 'Halfnorm', 'Uniform', 'UniformIncrease'])  # nopep8
            else:
                params.rule_generation__mutation = trial.suggest_categorical('rule_generation__mutation__mutation', ['Normal', 'Halfnorm', 'HalfnormIncrease', 'Uniform', 'UniformIncrease'])  # nopep8
            params.rule_generation__mutation = getattr(suprb.optimizer.rule.mutation, params.rule_generation__mutation)()  # nopep8

            if isinstance(params.rule_generation__mutation, suprb.optimizer.rule.mutation.SigmaRange):
                params.rule_generation__mutation__mutation = trial.suggest_categorical('rule_generation__mutation__mutation', ['Normal', 'Halfnorm', 'HalfnormIncrease', 'Uniform', 'UniformIncrease'])  # nopep8
                params.rule_generation__mutation__mutation = getattr(suprb.optimizer.rule.mutation, params.rule_generation__mutation__mutation)()  # nopep8

                params.rule_generation__mutation__low = trial.suggest_float('rule_generation__mutation__sigma_range__low', 0.001, 0.01)  # nopep8
                params.rule_generation__mutation__high = trial.suggest_float('rule_generation__mutation__sigma_range__high', 0.01, 0.1)  # nopep8

                params.rule_generation__mutation__mutation__sigma = trial.suggest_float('rule_generation__mutation__mutation__sigma', 0.05, 0.2)  # nopep8
            else:
                params.rule_generation__mutation__sigma = trial.suggest_float('rule_generation__mutation__sigma', 0.05, 0.2)  # nopep8

            # Origin Generation
            params.rule_generation__origin_generation = trial.suggest_categorical('rule_generation__origin_generation', ['UniformInputOrigin', 'UniformSamplesOrigin', 'Matching', 'SquaredError'])  # nopep8
            params.rule_generation__origin_generation = getattr(suprb.optimizer.rule.origin, params.rule_generation__origin_generation)()  # nopep8

            if isinstance(params.rule_generation__mutation, suprb.optimizer.rule.origin.Matching) or \
               isinstance(params.rule_generation__mutation, suprb.optimizer.rule.origin.SquaredError):
                params.rule_generation__origin_generation__use_elitist = trial.suggest_categorical('rule_generation__origin_generation__use_elitist', [True, False])  # nopep8

        params.solution_composition = trial.suggest_categorical('solution_composition', ['GeneticAlgorithm', 'ArtificialBeeColonyAlgorithm', 'AntColonyOptimization', 'GreyWolfOptimizer', 'ParticleSwarmOptimization', "RandomSearch"])  # nopep8

        if params.solution_composition == 'GeneticAlgorithm':
            # GA base
            params.solution_composition = getattr(suprb.optimizer.solution.ga, params.solution_composition)()

            params.solution_composition__n_iter = trial.suggest_int('solution_composition__n_iter', 16, 64)
            params.solution_composition__population_size = trial.suggest_int(
                'solution_composition__population_size', 16, 64)
            params.solution_composition__elitist_ratio = trial.suggest_float(
                'solution_composition__elitist_ratio', 0.0, 0.3)

            # GA init
            params.solution_composition__init = trial.suggest_categorical('solution_composition__init', ['ZeroInit', 'RandomInit'])  # nopep8
            params.solution_composition__init = getattr(
                suprb.solution.initialization, params.solution_composition__init)()

            if isinstance(params.solution_composition__init, suprb.solution.initialization.RandomInit):
                params.solution_composition__init__p = trial.suggest_float('solution_composition__init__p', 0.3, 0.8)

            params.solution_composition__init__fitness = trial.suggest_categorical('solution_composition__init__fitness', ['PseudoBIC', 'ComplexityEmary', 'ComplexityWu'])  # nopep8
            params.solution_composition__init__fitness = getattr(
                suprb.solution.fitness, params.solution_composition__init__fitness)()

            # Don't do this. It will kill your fitness because complexity = 0
            # if not isinstance(params.solution_composition__init__fitness, suprb.solution.fitness.PseudoBIC):
            #     params.solution_composition__init__fitness__alpha = trial.suggest_float('solution_composition__init__fitness__alpha', 0.0, 1.0) # nopep8

            params.solution_composition__init__mixing__experience_weight = trial.suggest_float(
                'solution_composition__init__mixing__experience_weight', 0.0, 1.0)

            params.solution_composition__init__mixing__experience_calculation = trial.suggest_categorical('solution_composition__init__mixing__experience_calculation', ['ExperienceCalculation', 'CapExperience', 'CapExperienceWithDimensionality'])  # nopep8
            params.solution_composition__init__mixing__experience_calculation = getattr(suprb.solution.mixing_model, params.solution_composition__init__mixing__experience_calculation)()  # nopep8

            if isinstance(params.solution_composition__init__mixing__experience_calculation, suprb.solution.mixing_model.CapExperienceWithDimensionality):
                params.solution_composition__init__mixing__experience_calculation__upper_bound = trial.suggest_float('solution_composition__init__mixing__experience_calculation__upper_bound', 2, 5)  # nopep8
            else:
                params.solution_composition__init__mixing__experience_calculation__upper_bound = trial.suggest_float('solution_composition__init__mixing__experience_calculation__upper_bound', 20, 50)  # nopep8

            params.solution_composition__init__mixing__filter_subpopulation = trial.suggest_categorical('solution_composition__init__mixing__filter_subpopulation', ['FilterSubpopulation', 'NBestFitness', 'NRandom', 'RouletteWheel'])  # nopep8
            params.solution_composition__init__mixing__filter_subpopulation = getattr(suprb.solution.mixing_model, params.solution_composition__init__mixing__filter_subpopulation)()  # nopep8

            params.solution_composition__init__mixing__filter_subpopulation__rule_amount = trial.suggest_int('solution_composition__init__mixing__filter_subpopulation__rule_amount', 4, 10)  # nopep8

            # GA selection
            params.solution_composition__selection = trial.suggest_categorical('solution_composition__selection', ['Random', 'RouletteWheel', 'LinearRank', 'Tournament'])  # nopep8
            params.solution_composition__selection = getattr(suprb.optimizer.solution.ga.selection, params.solution_composition__selection)()  # nopep8

            if isinstance(params.solution_composition__selection, suprb.optimizer.solution.ga.selection.Tournament):
                params.solution_composition__selection__k = trial.suggest_int('solution_composition__selection__k', 3, 10)  # nopep8

            params.solution_composition__mutation__mutation_rate = trial.suggest_float('solution_composition__mutation__mutation_rate', 0.0, 0.1)  # nopep8

            # GA crossover
            params.solution_composition__crossover = trial.suggest_categorical('solution_composition__crossover', ['NPoint', 'Uniform'])  # nopep8
            params.solution_composition__crossover = getattr(suprb.optimizer.solution.ga.crossover, params.solution_composition__crossover)()  # nopep8

            params.solution_composition__crossover__crossover_rate = trial.suggest_float('solution_composition__crossover__crossover_rate', 0.7, 1.0)  # nopep8
            if isinstance(params.solution_composition__crossover__crossover_rate, suprb.optimizer.solution.ga.crossover.NPoint):
                params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)  # nopep8

        elif params.solution_composition == 'ArtificialBeeColonyAlgorithm':
            params.solution_composition = getattr(suprb.optimizer.solution.abc, params.solution_composition)()

            params.solution_composition__food = trial.suggest_categorical(
                'solution_composition__food', ['Sigmoid', 'Bitwise', 'DimensionFlips'])
            params.solution_composition__food = getattr(
                suprb.optimizer.solution.abc.food, params.solution_composition__food)()

            params.solution_composition__trials_limit = trial.suggest_int('solution_composition__trials_limit', 1, 32)

            if isinstance(params.solution_composition__food, abc.food.DimensionFlips):
                params.solution_composition__food__flip_rate = trial.suggest_float(
                    'solution_composition__food__flip_rate', 0.01, 1)

        elif params.solution_composition == 'AntColonyOptimization':
            params.solution_composition = getattr(suprb.optimizer.solution.aco, params.solution_composition)()

            params.solution_composition__builder = trial.suggest_categorical(
                'solution_composition__builder', ['Binary', 'Complete'])
            params.solution_composition__builder = getattr(
                suprb.optimizer.solution.aco.builder, params.solution_composition__builder)()
            params.solution_composition__builder__alpha = trial.suggest_float(
                'solution_composition__builder__alpha', 0.5, 5)
            params.solution_composition__builder__beta = trial.suggest_float(
                'solution_composition__builder__beta', 0.5, 5)

            params.solution_composition__evaporation_rate = trial.suggest_float(
                'solution_composition__evaporation_rate', 0, 0.9)
            params.solution_composition__selection__n = trial.suggest_int(
                'solution_composition__selection__n', 1, 32 // 2)

        elif params.solution_composition == 'GreyWolfOptimizer':
            params.solution_composition = getattr(suprb.optimizer.solution.gwo, params.solution_composition)()

            params.solution_composition__position = trial.suggest_categorical(
                'solution_composition__position', ['Sigmoid', 'Crossover'])
            params.solution_composition__position = getattr(
                suprb.optimizer.solution.gwo.position, params.solution_composition__position)()
            params.solution_composition__n_leaders = trial.suggest_int('solution_composition__n_leaders', 1, 32 // 2)

        elif params.solution_composition == 'ParticleSwarmOptimization':
            params.solution_composition = getattr(suprb.optimizer.solution.pso, params.solution_composition)()

            params.solution_composition__movement = trial.suggest_categorical(
                'solution_composition__movement', ['Sigmoid', 'SigmoidQuantum', 'BinaryQuantum'])
            params.solution_composition__movement = getattr(
                suprb.optimizer.solution.pso.movement, params.solution_composition__movement)()

            params.solution_composition__a_min = trial.suggest_float('solution_composition__a_min', 0, 3)
            params.solution_composition__a_max = trial.suggest_float(
                'solution_composition__a_max', params.solution_composition__a_min, 3)

            if isinstance(params.solution_composition__movement, suprb.optimizer.solution.pso.movement.Sigmoid):
                params.solution_composition__movement__b = trial.suggest_float(
                    'solution_composition__movement__b', 0, 3)
                params.solution_composition__movement__c = trial.suggest_float(
                    'solution_composition__movement__c', 0, 3)
            elif isinstance(params.solution_composition__movement, suprb.optimizer.solution.pso.movement.BinaryQuantum):
                params.solution_composition__movement__p_learning = trial.suggest_float(
                    'solution_composition__movement__p_learning', 0.01, 1)
                params.solution_composition__movement__n_attractors = trial.suggest_int(
                    'solution_composition__movement__n_attractors', 1, 32 // 2)

        elif params.solution_composition == 'RandomSearch':
            params.solution_composition = getattr(suprb.optimizer.solution.rs, params.solution_composition)()

            params.solution_composition__n_iter = trial.suggest_int('solution_composition__n_iter', 64, 128)
            params.solution_composition__population_size = trial.suggest_int(
                'solution_composition__population_size', 64, 128)

        # print(estimator.get_params())
        # exit()

    experiment_name = f'SupRB Tuning j:{job_id} p:{problem}'
    print(experiment_name)
    experiment = Experiment(name=experiment_name,  verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    evaluation = CrossValidate(
        estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(
        n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
