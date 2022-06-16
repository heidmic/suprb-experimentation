import numpy as np
import click
import mlflow
from optuna import Trial
import suprb

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
from suprb.optimizer.rule import ns, origin
from suprb.optimizer.rule.mutation import HalfnormIncrease


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-t', '--ns_type', type=click.STRING, default='NS')
def run(problem: str, ns_type: str):
    print(f"{ns_type} is tuned with problem {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_generation=ns.NoveltySearch(
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                              model=Ridge(alpha=0.01,
                                                          random_state=random_state)),
            ns_type=ns_type,
            origin_generation=origin.SquaredError(),
            mutation=HalfnormIncrease()
        ),
        solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32),
        n_iter=32,
        n_rules=8,
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
        timeout=24 * 60 * 60,  # 24 hours
        scoring='neg_mean_squared_error',
        verbose=10
    )

    @param_space()
    def suprb_NS_GA_space(trial: Trial, params: Bunch):
        # NS
        # sigma_space = [0, 2]
        sigma_space = [0, np.sqrt(X.shape[1])]
        params.rule_generation__mutation__sigma = trial.suggest_float('mutation_sigma', *sigma_space)

        params.rule_generation__n_iter = trial.suggest_int('n_iter', 0, 20)
        params.rule_generation__mu = trial.suggest_int('mu', 7, 20)
        params.rule_generation__lm_ratio = trial.suggest_int('lm_ratio', 5, 15)

        params.rule_generation__selection = trial.suggest_categorical('selection',
                                                                      ['RouletteWheel', 'Random'])
        params.rule_generation__selection = getattr(suprb.optimizer.rule.selection, params.rule_generation__selection)()

        params.rule_generation__archive = trial.suggest_categorical('archive', ['novelty', 'random', 'none'])

        params.rule_generation__novelty_fitness_combination = \
            trial.suggest_categorical('novelty_fitness_combination',
                                      ['novelty', '50/50', '75/25',
                                       'pmcns', 'pareto'])

        if ns_type == 'MCNS':
            params.rule_generation__MCNS_threshold_matched = trial.suggest_int('MCNS_threshold_matched', 10, 20)
        elif ns_type == 'NSLC':
            params.rule_generation__NSLC_threshold = trial.suggest_int('rule_generation__NSLC_threshold', 10, 20)

        # GA
        params.solution_composition__selection = trial.suggest_categorical(
            'solution_composition__selection',
            ['RouletteWheel',
             'Tournament',
             'LinearRank', 'Random'])
        params.solution_composition__selection = getattr(ga.selection, params.solution_composition__selection)()

        if isinstance(params.solution_composition__selection, ga.selection.Tournament):
            params.solution_composition__selection__k = trial.suggest_int('solution_composition__selection__k', 3, 10)

        params.solution_composition__crossover = trial.suggest_categorical('solution_composition__crossover',
                                                                           ['NPoint', 'Uniform'])
        params.solution_composition__crossover = getattr(ga.crossover, params.solution_composition__crossover)()

        if isinstance(params.solution_composition__crossover, ga.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)

        params.solution_composition__mutation__mutation_rate = trial.suggest_float(
            'solution_composition__mutation_rate', 0, 0.1)

    experiment = Experiment(name=f'{problem} NS Tuning', verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_NS_GA_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment("NS Tuning")
    log_experiment(experiment)


if __name__ == '__main__':
    run()
