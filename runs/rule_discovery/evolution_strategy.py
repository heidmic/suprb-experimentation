import math

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
from runs.solution_composition.shared_config import load_dataset, random_state

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es, origin, mutation


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
def run(problem: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_generation=es.ES1xLambda(
            operator='&',
            n_iter=10_000,
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                              model=Ridge(alpha=0.01,
                                                          random_state=random_state)),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),
        ),
        solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32),
        n_iter=32,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger(
            [('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=10_000,
        timeout=90 * 60 * 60,  # 90 hours
        verbose=10
    )

    @param_space('suprb_ES_GA')
    def suprb_ES_GA_space(trial: Trial, params: Bunch):
        # ES
        sigma_space = [0, math.sqrt(X.shape[1])]

        params.mutation__sigma = trial.suggest_float('mutation__sigma', *sigma_space)
        params.delay = trial.suggest_int('delay', 10, 100)
        params.init__fitness__alpha = trial.suggest_float(
            'init__fitness__alpha', 0.01, 0.2)

        # GA
        params.selection = trial.suggest_categorical('selection',
                                                     ['RouletteWheel',
                                                      'Tournament',
                                                      'LinearRank', 'Random'])
        params.selection = getattr(ga.selection, params.selection)()

        if isinstance(params.selection, ga.selection.Tournament):
            params.selection__k = trial.suggest_int('selection__k', 3, 10)

        params.crossover = trial.suggest_categorical('crossover',
                                                     ['NPoint', 'Uniform'])
        params.crossover = getattr(ga.crossover, params.crossover)()

        if isinstance(params.crossover, ga.crossover.NPoint):
            params.crossover__n = trial.suggest_int('crossover__n', 1, 10)

        params.mutation__mutation_rate = trial.suggest_float('mutation_rate', 0,
                                                             0.1)

    experiment = Experiment(name=f'{problem} ES Tuning', verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_ES_GA_space, tuner=tuner)

    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment("ES Tuning")
    log_experiment(experiment)


if __name__ == '__main__':
    run()


