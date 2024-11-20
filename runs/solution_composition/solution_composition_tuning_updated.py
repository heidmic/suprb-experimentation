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
from suprb.optimizer.rule import es, origin, mutation
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd


from experiments import Experiment
from experiments.mlflow import log_experiment
from experiments.parameter_search import solution_composition_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)
    else:
        enc = LabelEncoder()
        dataset = fetch_openml(name=name, version=1)

        if name == "meta":
            dataset.data.DS_Name = enc.fit_transform(dataset.data.DS_Name)
            dataset.data.Alg_Name = enc.fit_transform(dataset.data.Alg_Name)
            dataset.data = dataset.data.drop(
                dataset.data.columns[dataset.data.isna().any()].tolist(), axis=1)

        if name == "chscase_foot":
            dataset.data.col_1 = enc.fit_transform(dataset.data.col_1)

        if isinstance(dataset.data, np.ndarray):
            X = dataset.data
        elif isinstance(dataset.data, pd.DataFrame) or isinstance(dataset.data, pd.Series):
            X = dataset.data.to_numpy(dtype=float)
        else:
            X = dataset.data.toarray()

        if isinstance(dataset.target, np.ndarray):
            y = dataset.target
        elif isinstance(dataset.target, pd.DataFrame) or isinstance(dataset.target, pd.Series):
            y = dataset.target.to_numpy(dtype=float)
        else:
            y = dataset.target.toarray()

        return X, y


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='concrete_strength')
@click.option('-o', '--optimizer', type=click.STRING, default='GeneticAlgorithm')
def run(problem: str, optimizer: str):
    print(f"Problem is {problem}, optimizer is {optimizer}")

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
        timeout=72 * 60 * 60,  # 72 hours
        scoring='fitness',
        verbose=10
    )

    @param_space()
    def suprb_space(trial: Trial, params: Bunch):
        # ES
        sigma_space = [0, np.sqrt(X.shape[1])]

        params.rule_generation__mutation__sigma = trial.suggest_float(
            'rule_generation__mutation__sigma', *sigma_space)
        params.rule_generation__delay = trial.suggest_int('rule_generation__delay', 10, 200)
        params.rule_generation__init__fitness__alpha = trial.suggest_float(
            'rule_generation__init__fitness__alpha', 0.01, 0.2)

        params.solution_composition = optimizer  # trial.suggest_categorical('solution_composition', ['GeneticAlgorithm', 'ArtificialBeeColonyAlgorithm', 'AntColonyOptimization', 'GreyWolfOptimizer', 'ParticleSwarmOptimization', "RandomSearch"])  # nopep8

        if params.solution_composition == 'GeneticAlgorithm':
            # GA base
            params.solution_composition = getattr(suprb.optimizer.solution.ga, params.solution_composition)()

            params.solution_composition__elitist_ratio = trial.suggest_float('solution_composition__elitist_ratio', 0.0, 0.3)

            # GA init
            params.solution_composition__init = trial.suggest_categorical('solution_composition__init', ['ZeroInit', 'RandomInit'])  # nopep8
            params.solution_composition__init = getattr(suprb.solution.initialization, params.solution_composition__init)()

            if isinstance(params.solution_composition__init, suprb.solution.initialization.RandomInit):
                params.solution_composition__init__p = trial.suggest_float('solution_composition__init__p', 0.3, 0.8)

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
            params.solution_composition__food = getattr(suprb.optimizer.solution.abc.food, params.solution_composition__food)()

            params.solution_composition__trials_limit = trial.suggest_int('solution_composition__trials_limit', 1, 32)

            if isinstance(params.solution_composition__food, abc.food.DimensionFlips):
                params.solution_composition__food__flip_rate = trial.suggest_float('solution_composition__food__flip_rate', 0.01, 1)

        elif params.solution_composition == 'AntColonyOptimization':
            params.solution_composition = getattr(suprb.optimizer.solution.aco, params.solution_composition)()

            params.solution_composition__builder = trial.suggest_categorical('solution_composition__builder', ['Binary', 'Complete'])
            params.solution_composition__builder = getattr(suprb.optimizer.solution.aco.builder, params.solution_composition__builder)()
            params.solution_composition__builder__alpha = trial.suggest_float('solution_composition__builder__alpha', 0.5, 5)
            params.solution_composition__builder__beta = trial.suggest_float('solution_composition__builder__beta', 0.5, 5)

            params.solution_composition__evaporation_rate = trial.suggest_float('solution_composition__evaporation_rate', 0, 0.9)
            params.solution_composition__selection__n = trial.suggest_int('solution_composition__selection__n', 1, 32 // 2)

        elif params.solution_composition == 'GreyWolfOptimizer':
            params.solution_composition = getattr(suprb.optimizer.solution.gwo, params.solution_composition)()

            params.solution_composition__position = trial.suggest_categorical('solution_composition__position', ['Sigmoid', 'Crossover'])
            params.solution_composition__position = getattr(suprb.optimizer.solution.gwo.position, params.solution_composition__position)()
            params.solution_composition__n_leaders = trial.suggest_int('solution_composition__n_leaders', 1, 32 // 2)

        elif params.solution_composition == 'ParticleSwarmOptimization':
            params.solution_composition = getattr(suprb.optimizer.solution.pso, params.solution_composition)()

            params.solution_composition__movement = trial.suggest_categorical(
                'solution_composition__movement', ['Sigmoid', 'SigmoidQuantum', 'BinaryQuantum'])
            params.solution_composition__movement = getattr(suprb.optimizer.solution.pso.movement, params.solution_composition__movement)()

            params.solution_composition__a_min = trial.suggest_float('solution_composition__a_min', 0, 3)
            params.solution_composition__a_max = trial.suggest_float('solution_composition__a_max', params.solution_composition__a_min, 3)

            if isinstance(params.solution_composition__movement, suprb.optimizer.solution.pso.movement.Sigmoid):
                params.solution_composition__movement__b = trial.suggest_float('solution_composition__movement__b', 0, 3)
                params.solution_composition__movement__c = trial.suggest_float('solution_composition__movement__c', 0, 3)
            elif isinstance(params.solution_composition__movement, suprb.optimizer.solution.pso.movement.BinaryQuantum):
                params.solution_composition__movement__p_learning = trial.suggest_float(
                    'solution_composition__movement__p_learning', 0.01, 1)
                params.solution_composition__movement__n_attractors = trial.suggest_int(
                    'solution_composition__movement__n_attractors', 1, 32 // 2)

        elif params.solution_composition == 'RandomSearch':
            params.solution_composition = getattr(suprb.optimizer.solution.rs, params.solution_composition)()

            params.solution_composition__n_iter = trial.suggest_int('solution_composition__n_iter', 64, 128)
            params.solution_composition__population_size = trial.suggest_int('solution_composition__population_size', 64, 128)

    experiment_name = f'SupRB Tuning o:{optimizer} p:{problem}'
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
