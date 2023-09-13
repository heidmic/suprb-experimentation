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
from suprb.optimizer.solution import ga, saga1, saga2, saga3, sas
from suprb.optimizer.rule import es
from suprb.solution.initialization import RandomInit
import suprb.solution.mixing_model as mixing_model
import suprb


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def validate_solution_composition(name: str):
    valid_names = ['ga', 'saga1', 'saga2', 'saga3', 'sas']
    if not name in valid_names:
        raise ValueError('Invalid solution composition')


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-s', '--solution_composition', type=click.STRING, default='ga')
@click.option('-j', '--job_id', type=click.STRING, default='NA')
def run(problem: str, solution_composition: str, job_id: str):
    print(f"Problem is {problem}, with solution composition {solution_composition} and job id {job_id}")

    validate_solution_composition(solution_composition)
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(
        rule_generation=es.ES1xLambda(),
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
        timeout=60*60*72,  # 72 hours
        scoring='neg_mean_squared_error',
        verbose=10
    )
    

    @param_space()
    def suprb_space(trial: Trial, params: Bunch):
        # ES base
        params.rule_generation__operator = trial.suggest_categorical('rule_generation__operator', ['&', ',', '+'])  # nopep8

        params.rule_generation__lmbda = trial.suggest_int('rule_generation__lmbda', 10, 30)  # nopep8
        params.rule_generation__n_iter = trial.suggest_int('rule_generation__n_iter', 500, 1500)  # nopep8

        if params.rule_generation__operator == '&':
            params.rule_generation__delay = trial.suggest_int('rule_generation__delay', 20, 50)  # nopep8

        # Acceptance
        params.rule_generation__init = trial.suggest_categorical('rule_generation__init', ['MeanInit', 'NormalInit', 'HalfnormInit'])  # nopep8
        params.rule_generation__init = getattr(suprb.rule.initialization, params.rule_generation__init)()  # nopep8

        if not isinstance(params.rule_generation__init, suprb.rule.initialization.MeanInit):
            params.rule_generation__init__sigma = trial.suggest_float('rule_generation__init__sigma', 0.01, 0.1)  # nopep8

        params.rule_generation__init__fitness = params.rule_generation__init__fitness = suprb.rule.fitness.VolumeWu()
        params.rule_generation__init__fitness__alpha = trial.suggest_float('rule_generation__init__fitness__alpha', 0.5, 1)  # nopep8

        params.rule_generation__init__model = Ridge()

        # Mutation 
        params.rule_generation__mutation = trial.suggest_categorical('rule_generation__mutation', ['Normal', 'HalfnormIncrease', 'UniformIncrease'])  # nopep8
        params.rule_generation__mutation = getattr(suprb.optimizer.rule.mutation, params.rule_generation__mutation)()  # nopep8
        params.rule_generation__mutation__sigma = trial.suggest_float('rule_generation__mutation__sigma', 0.0, 3.0)  # nopep8

       # Solution Composition
        if solution_composition == 'ga':
            # GA base
            params.solution_composition = ga.GeneticAlgorithm()

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
        
        elif solution_composition == 'saga1':
            # SAGA1 Base
            params.solution_composition = saga1.SelfAdaptingGeneticAlgorithm()

            params.solution_composition__n_iter = trial.suggest_int('solution_composition__n_iter', 16, 128, step=2)

            # SAGA1 selection
            params.solution_composition__selection = trial.suggest_categorical('solution_composition__selection', ['Random', 'RouletteWheel', 'LinearRank', 'Tournament'])  # nopep8
            params.solution_composition__selection = getattr(suprb.optimizer.solution.saga1.selection, params.solution_composition__selection)()  # nopep8

            if isinstance(params.solution_composition__selection, suprb.optimizer.solution.saga1.selection.Tournament):
                params.solution_composition__selection__k = trial.suggest_int('solution_composition__selection__k', 3, 10)  # nopep8

            # SAGA1 crossover
            params.solution_composition__crossover = trial.suggest_categorical('solution_composition__crossover', ['NPoint', 'Uniform'])  # nopep8
            params.solution_composition__crossover = getattr(suprb.optimizer.solution.saga1.crossover, params.solution_composition__crossover)()  # nopep8

            if isinstance(params.solution_composition__crossover, suprb.optimizer.solution.saga1.crossover.NPoint):
                params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)  # nopep8

        elif solution_composition == 'saga2':
            # SAGA2 Base
            params.solution_composition = saga2.SelfAdaptingGeneticAlgorithm()

            params.solution_composition__n_iter = trial.suggest_int('solution_composition__n_iter', 16, 128, step=2)

            # SAGA2 selection
            params.solution_composition__selection = trial.suggest_categorical('solution_composition__selection', ['Random', 'RouletteWheel', 'LinearRank', 'Tournament'])  # nopep8
            params.solution_composition__selection = getattr(suprb.optimizer.solution.saga2.selection, params.solution_composition__selection)()  # nopep8

            if isinstance(params.solution_composition__selection, suprb.optimizer.solution.saga2.selection.Tournament):
                params.solution_composition__selection__k = trial.suggest_int('solution_composition__selection__k', 3, 10)  # nopep8

            # SAGA2 crossover
            params.solution_composition__crossover = trial.suggest_categorical('solution_composition__crossover', ['NPoint', 'Uniform'])  # nopep8
            params.solution_composition__crossover = getattr(suprb.optimizer.solution.saga2.crossover, params.solution_composition__crossover)()  # nopep8

            if isinstance(params.solution_composition__crossover, suprb.optimizer.solution.saga2.crossover.NPoint):
                params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)  # nopep8

        elif solution_composition == 'saga3':
            # SAGA3 Base
            params.solution_composition = saga3.SelfAdaptingGeneticAlgorithm()

            params.solution_composition__n_iter = trial.suggest_int('solution_composition__n_iter', 16, 128, step=2)

            # SAGA3 selection
            params.solution_composition__selection = trial.suggest_categorical('solution_composition__selection', ['Random', 'RouletteWheel', 'LinearRank', 'Tournament'])  # nopep8
            params.solution_composition__selection = getattr(suprb.optimizer.solution.saga3.selection, params.solution_composition__selection)()  # nopep8

            if isinstance(params.solution_composition__selection, suprb.optimizer.solution.saga3.selection.Tournament):
                params.solution_composition__selection__k = trial.suggest_int('solution_composition__selection__k', 3, 10)  # nopep8

            # SAGA3 crossover
            # not needed as it is selfadapting

        elif solution_composition == 'sas':
            # SAS Base
            params.solution_composition = sas.SasGeneticAlgorithm()

            params.solution_composition__n_iter = trial.suggest_int('solution_composition__n_iter', 16, 128, step=2)

            # SAS selection
            # Not needed as it uses a selfadapting ageing selection

            # SAS crossover
            params.solution_composition__crossover = trial.suggest_categorical('solution_composition__crossover', ['NPoint', 'Uniform'])  # nopep8
            params.solution_composition__crossover = getattr(suprb.optimizer.solution.sas.crossover, params.solution_composition__crossover)()  # nopep8

            params.solution_composition__crossover__crossover_rate = trial.suggest_float('solution_composition__crossover__crossover_rate', 0.7, 1.0)  # nopep8
            if isinstance(params.solution_composition__crossover, suprb.optimizer.solution.sas.crossover.NPoint):
                params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)  # nopep8

            params.solution_composition__mutation__mutation_rate = trial.suggest_float('solution_composition__mutation__mutation_rate', 0.0, 0.1)  # nopep8




        # print(estimator.get_params())
        # exit()
        
        
        

    experiment_name = f'SupRB Tuning j:{job_id} p:{problem} s:{solution_composition}'
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