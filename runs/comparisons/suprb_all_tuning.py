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
from suprb.optimizer.solution import ga
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
def run(problem: str, job_id: str):
    print(f"Problem is {problem}, with job id {job_id}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_generation=rs.RandomSearch(),
        solution_composition=ga.GeneticAlgorithm(
            n_iter=32, population_size=32, selection=ga.selection.Tournament()),
        n_iter=32, n_rules=4, verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()),
                               ('default', DefaultLogger())]),)

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=1000,
        timeout=60*60*24,  # 24 hours
        scoring='neg_mean_squared_error',
        verbose=10
    )
    # print(estimator.get_params())
    # exit()

    @param_space()
    def suprb_space(trial: Trial, params: Bunch):
        params.rule_generation = trial.suggest_categorical('rule_generation', ['ES1xLambda', 'NoveltySearch', 'RandomSearch'])  # nopep8

        if params.rule_generation == 'ES1xLambda':
            # ES base
            params.rule_generation = getattr(suprb.optimizer.rule.es, params.rule_generation)()

            params.rule_generation__operator = trial.suggest_categorical('rule_generation__operator', ['&', ',', '+'])

            params.rule_generation__lmbda = trial.suggest_int('rule_generation__lmbda', 10, 30)
            params.rule_generation__n_jobs = trial.suggest_int('rule_generation__n_jobs', 1, 5)
            params.rule_generation__n_iter = trial.suggest_int('rule_generation__n_iter', 500, 1500)

            if params.rule_generation__operator == '&':
                params.rule_generation__delay = trial.suggest_int('rule_generation__delay', 20, 50)
        elif params.rule_generation == 'NoveltySearch':
            # NS base
            params.rule_generation = getattr(suprb.optimizer.rule.ns, params.rule_generation)()

            # NS base
            params.rule_generation__lmbda = trial.suggest_int('rule_generation__lmbda', 100, 200)
            params.rule_generation__mu = trial.suggest_int('rule_generation__mu', 10, 20)
            params.rule_generation__n_elitists = trial.suggest_int('rule_generation__n_elitists', 5, 20)
            params.rule_generation__n_iter = trial.suggest_int('rule_generation__n_iter', 5, 15)
            params.rule_generation__n_jobs = trial.suggest_int('rule_generation__n_jobs', 1, 5)
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

            # NS crossover (TODO: We only have UniformCrossover. Is this correct? matching_type is None here as well. Is that intended?)
            params.rule_generation__crossover__crossover_rate = trial.suggest_float('rule_generation__crossover__crossover_rate', 0.1, 0.5)  # nopep8
        elif params.rule_generation == 'RandomSearch':
            # RS base
            params.rule_generation = getattr(suprb.optimizer.rule.rs, params.rule_generation)()

            params.rule_generation__n_jobs = trial.suggest_int('rule_generation__n_jobs', 1, 5)
            params.rule_generation__n_iter = trial.suggest_int('rule_generation__n_iter', 1, 5)
            params.rule_generation__rules_generated = trial.suggest_int('rule_generation__rules_generated', 5, 10)

        # Acceptance
        params.rule_generation__acceptance = trial.suggest_categorical('rule_generation__acceptance', ['Variance', 'MaxError'])  # nopep8
        params.rule_generation__acceptance = getattr(suprb.optimizer.rule.acceptance, params.rule_generation__acceptance)()  # nopep8

        if isinstance(params.rule_generation__acceptance, suprb.optimizer.rule.acceptance.Variance):
            params.rule_generation__acceptance__beta = trial.suggest_float('rule_generation__acceptance__beta', 1, 3)

        # Constraint (TODO: Do we need to specify min_range and clip or can we use default values here?)
        params.rule_generation__constraint = trial.suggest_categorical('rule_generation__constraint', ['MinRange', 'Clip', 'CombinedConstraint'])  # nopep8
        params.rule_generation__constraint = getattr(suprb.optimizer.rule.constraint, params.rule_generation__constraint)()  # nopep8

        if isinstance(params.rule_generation__constraint, suprb.optimizer.rule.constraint.CombinedConstraint):
            params.rule_generation__constraint__clip = suprb.optimizer.rule.constraint.Clip()
            params.rule_generation__constraint__min_range = suprb.optimizer.rule.constraint.MinRange()

        # Init (TODO: bounds and matching_typeare None and are apparenly never said. Is this intended?)
        params.rule_generation__init = trial.suggest_categorical('rule_generation__init', ['MeanInit', 'NormalInit', 'HalfnormInit'])  # nopep8
        params.rule_generation__init = getattr(suprb.rule.initialization, params.rule_generation__init)()  # nopep8

        if not isinstance(params.rule_generation__init, suprb.rule.initialization.MeanInit):
            params.rule_generation__init__sigma = trial.suggest_float('rule_generation__init__sigma', 0.05, 0.2)  # nopep8

        params.rule_generation__init__fitness = trial.suggest_categorical('rule_generation__init__fitness', ['PseudoAccuracy', 'VolumeEmary', 'VolumeWu'])  # nopep8
        params.rule_generation__init__fitness = getattr(suprb.rule.fitness, params.rule_generation__init__fitness)()  # nopep8

        if not isinstance(params.rule_generation__init__fitness, suprb.rule.fitness.PseudoAccuracy):
            params.rule_generation__init__fitness__alpha = trial.suggest_float('rule_generation__init__fitness__alpha', 0.5, 1)  # nopep8

        # TODO: What the we want to tune besides Ridge? And do we want to tune sklearn params as well (alpha, copy_X, fit_intercept, max_iter, normalize, positive, random_state, solver, tol)
        params.rule_generation__init__model = Ridge()

        # Selection
        params.rule_generation__selection = trial.suggest_categorical('rule_generation__selection', ['Fittest', 'RouletteWheel', 'NondominatedSort', 'Random'])  # nopep8
        params.rule_generation__selection = getattr(suprb.optimizer.rule.selection, params.rule_generation__selection)()  # nopep8

        if isinstance(params.rule_generation, suprb.optimizer.rule.es.ES1xLambda) or \
           isinstance(params.rule_generation, suprb.optimizer.rule.ns.NoveltySearch):
            # Mutation (# TODO: Do we also need matching_type here? It is always None, is that expected?)
            params.rule_generation__mutation = trial.suggest_categorical('rule_generation__mutation', ['SigmaRange', 'Normal', 'Halfnorm', 'HalfnormIncrease', 'Uniform', 'UniformIncrease'])  # nopep8
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

        print(params)

        # "matching_type": "None",
        # "n_initial_rules": 0,
        # "n_iter": 32,
        # "n_jobs": 1,
        # "n_rules": 4,
        # "random_state": "None",

        # n_iter=1000,
        # "mutation=HalfnormIncrease()",
        # "origin_generation=SquaredError())",
        # "solution_composition__archive": "Elitist()",
        # "solution_composition__crossover__crossover_rate": 0.9,
        # "solution_composition__crossover__n": 3,
        # "solution_composition__crossover": NPoint(n = 3),
        # "solution_composition__elitist_ratio": 0.17,
        # "solution_composition__init__fitness__alpha": 0.3,
        # "solution_composition__init__fitness": "ComplexityWu()",
        # "solution_composition__init__mixing__experience_calculation": < suprb.solution.mixing_model.ExperienceCalculation object at 0x7fd7db3f6850 > ,
        # "solution_composition__init__mixing__experience_weight": 1,
        # "solution_composition__init__mixing__filter_subpopulation": < suprb.solution.mixing_model.FilterSubpopulation object at 0x7fd7db3f6610 > ,
        # "solution_composition__init__mixing": "ErrorExperienceHeuristic()",
        # "solution_composition__init__p": 0.5,
        # "solution_composition__init": "RandomInit(fitness=ComplexityWu()",
        # "mixing=ErrorExperienceHeuristic())",
        # "solution_composition__mutation__mutation_rate": 0.001,
        # "solution_composition__mutation": BitFlips(mutation_rate = 0.001),
        # "solution_composition__n_iter": 32,
        # "solution_composition__n_jobs": 1,
        # "solution_composition__population_size": 32,
        # "solution_composition__random_state": "None",
        # "solution_composition__selection__k": 5,
        # "solution_composition__selection": "Tournament()",
        # "solution_composition__warm_start": true,
        # "solution_composition": "GeneticAlgorithm()",
        # "verbose": 10

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
