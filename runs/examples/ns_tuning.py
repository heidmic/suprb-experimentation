import mlflow
import numpy as np
import suprb2.rule.initialization
from optuna import Trial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from skopt.space import Integer, Real
from suprb2 import optimizer, SupRB2, rule
from suprb2.logging.combination import CombinedLogger
from suprb2.logging.default import DefaultLogger
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.rule import ns

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from experiments.parameter_search.skopt import SkoptTuner
from problems import scale_X_y
from problems.datasets import load_airfoil_self_noise
from sklearn.utils import Bunch, shuffle
import click
import optuna
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.model_selection import ShuffleSplit
from suprb2.optimizer.solution import ga

random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='gas_turbine')
def run(problem: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB2(
        rule_generation=ns.NoveltySearch(
            init=suprb2.rule.initialization.HalfnormInit()
        ),
        solution_composition=optimizer.solution.ga.GeneticAlgorithm(),
        n_iter=32,
        n_rules=8,
        verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )

    # cross_validate(estimator, X, y, scoring=mean_squared_error, verbose=10)

    # exit()

    n_calls = 128
    shared_tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=n_calls,
        timeout=24 * 60 * 60 * 4,  # 96 hours
        verbose=10
    )

    tuner = OptunaTuner(X_train=X, y_train=y,
                        scoring='neg_mean_squared_error',
                        **shared_tuning_params)

    # Create the base experiment, using some default tuner
    experiment = Experiment(name='NoveltySearch', verbose=10)

    @param_space()
    def optuna_objective(trial: optuna.Trial, params: Bunch):
        sigma_space = [0, 2]

        params.rule_generation__n_iter = trial.suggest_int('n_iter', 1, 100)
        params.rule_generation__mu = trial.suggest_int('mu', 8, 32)
        params.rule_generation__lm_ratio = trial.suggest_int('lm_ratio', 1, 64)

        params.rule_generation__origin_generation = trial.suggest_categorical('origin_generation',
                                                                              ['UniformSamplesOrigin',
                                                                               'Matching',
                                                                               'SquaredError'])
        params.rule_generation__origin_generation = getattr(optimizer.rule.origin,
                                                            params.rule_generation__origin_generation)()

        # params.rule_generation__init = trial.suggest_categorical('init', ['MeanInit', 'NormalInit', 'HalfnormInit'])
        # params.rule_generation__init = getattr(rule.initialization, params.rule_generation__init)()

        params.rule_generation__mutation__sigma = trial.suggest_float('mutation_sigma', *sigma_space)
        params.rule_generation__mutation = trial.suggest_categorical('mutation',
                                                                     ['Normal', 'Halfnorm',
                                                                      'HalfnormIncrease', 'Uniform',
                                                                      'UniformIncrease', ])
        params.rule_generation__mutation = getattr(optimizer.rule.mutation, params.rule_generation__mutation)()

        # crossover rate müsste noch implementiert werden, n point existiert auch für Rule Generation noch nicht

        # Tournament existiert nicht für Rule Generation
        params.rule_generation__selection = trial.suggest_categorical('selection',
                                                                      ['RouletteWheel', 'Random'])
        params.rule_generation__selection = getattr(optimizer.rule.selection, params.rule_generation__selection)()

        params.rule_generation__ns_type = trial.suggest_categorical('ns_type', ['NS', 'NSLC', 'MCNS'])

        if params.rule_generation__ns_type == 'MCNS':
            params.rule_generation__MCNS_threshold_matched = trial.suggest_int('MCNS_threshold_matched', 1, 100)

        params.rule_generation__archive = trial.suggest_categorical('archive', ['novelty', 'random', 'none'])

        params.rule_generation__novelty_fitness_combination = trial.suggest_categorical('novelty_fitness_combination',
                                                                                        ['novelty', '50/50', '75/25',
                                                                                         'pmcns', 'pareto'])

        # GA
        params.solution_composition__selection = trial.suggest_categorical('ga_selection',
                                                     ['RouletteWheel', 'Tournament', 'LinearRank', 'Random'])

        params.solution_composition__selection = getattr(ga.selection, params.solution_composition__selection)()

        if isinstance(params.solution_composition__selection, ga.selection.Tournament):
            params.solution_composition__selection__k = trial.suggest_int('ga_selection__k', 3, 10)

        params.solution_composition__crossover = trial.suggest_categorical('ga_crossover', ['NPoint', 'Uniform'])
        params.solution_composition__crossover = getattr(ga.crossover, params.solution_composition__crossover)()

        if isinstance(params.solution_composition__crossover, ga.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int('ga_crossover__n', 1, 10)

        params.solution_composition__mutation__mutation_rate = trial.suggest_float('mutation_rate', 0, 0.1)

    experiment.with_tuning(optuna_objective, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=4)

    # Evaluation using cross-validation and an external test set
    evaluation = CrossValidate(estimator=estimator, X=X, y=y,
                               random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(f"{problem}_opt{n_calls}x")
    log_experiment(experiment)


if __name__ == '__main__':
    run()
