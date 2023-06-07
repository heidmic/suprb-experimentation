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
from suprb.solution.initialization import RandomInit
import suprb.solution.mixing_model as mixing_model


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
        rule_generation=es.ES1xLambda(
            operator='&', n_iter=150, init=rule.initialization.MeanInit(
                fitness=rule.fitness.VolumeWu(),
                model=Ridge(alpha=0.01, random_state=random_state)),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),),
        solution_composition=ga.GeneticAlgorithm(
            n_iter=32, population_size=32,
            init=RandomInit(
                mixing=mixing_model.ErrorExperienceHeuristic(
                    filter_subpopulation=mixing_model.RouletteWheel(2, 2),
                    experience_calculation=mixing_model.CapExperienceWithDimensionality(10, 15),
                    experience_weight=2))),
        n_iter=2, n_rules=4, verbose=10,
        logger=CombinedLogger([('stdout', StdoutLogger()),
                               ('default', DefaultLogger())]),)

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=1,
        n_jobs=1,
        n_calls=2,
        timeout=60,  # 72 hours
        scoring='neg_mean_squared_error',
        verbose=10
    )

    @param_space()
    def suprb_ES_GA_space(trial: Trial, params: Bunch):
        # ES
        sigma_space = [0, np.sqrt(X.shape[1])]

        params.rule_generation__mutation__sigma = trial.suggest_float(
            'rule_generation__mutation__sigma', *sigma_space)
        params.rule_generation__init__fitness__alpha = trial.suggest_float(
            'rule_generation__init__fitness__alpha', 0.01, 0.2)
        # params.rule_generation__operator = trial.suggest_categorical(
        #    'rule_generation__operator', ['&', ',', '+'])

        # if params.rule_generation__operator == '&':
        #    params.rule_generation__delay = trial.suggest_int('rule_generation__delay', 10, 100)

        # params.rule_generation__mutation = trial.suggest_categorical(
        #    'mutation', ['Normal', 'HalfnormIncrease', 'UniformIncrease'])
        # params.rule_generation__mutation = getattr(mutation, params.rule_generation__mutation)()

        # GA
        params.solution_composition__selection = trial.suggest_categorical(
            'solution_composition__selection',
            ['RouletteWheel',
             'Tournament',
             'LinearRank', 'Random'])
        params.solution_composition__selection = getattr(
            ga.selection, params.solution_composition__selection)()

        if isinstance(params.solution_composition__selection, ga.selection.Tournament):
            params.solution_composition__selection__k = trial.suggest_int(
                'solution_composition__selection__k', 3, 10)

        params.solution_composition__crossover = trial.suggest_categorical('solution_composition__crossover',
                                                                           ['NPoint', 'Uniform'])
        params.solution_composition__crossover = getattr(
            ga.crossover, params.solution_composition__crossover)()

        if isinstance(params.solution_composition__crossover, ga.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int(
                'solution_composition__crossover__n', 1, 10)

        params.solution_composition__mutation__mutation_rate = trial.suggest_float(
            'solution_composition__mutation_rate', 0, 0.1)

        params.solution_composition__init__mixing__filter_subpopulation__rule_amount = trial.suggest_float(
            'solution_composition__init__mixing__filter_subpopulation__rule_amount', 10, 20)

        params.solution_composition__init__mixing__experience_weight = trial.suggest_float(
            'solution_composition__init__mixing__experience_weight', 0, 2)
        params.solution_composition__init__mixing__experience_calculation__lower_bound = trial.suggest_float(
            'solution_composition__init__mixing__experience_calculation__lower_bound', 0, 10)
        params.solution_composition__init__mixing__experience_calculation__upper_bound = trial.suggest_float(
            'solution_composition__init__mixing__experience_calculation__upper_bound', 20, 50)

        params.solution_composition__init__mixing__filter_subpopulation = trial.suggest_categorical(
            'solution_composition__init__mixing__filter_subpopulation',
            ['FilterSubpopulation', 'NBestFitness', 'NRandom', 'RouletteWheel'])
        params.solution_composition__init__mixing__filter_subpopulation = getattr(
            mixing_model, params.solution_composition__init__mixing__filter_subpopulation)()

        params.solution_composition__init__mixing__experience_calculation = trial.suggest_categorical(
            'solution_composition__init__mixing__experience_calculation',
            ['ExperienceCalculation', 'CapExperience', 'CapExperienceWithDimensionality'])
        params.solution_composition__init__mixing__experience_calculation = getattr(
            mixing_model, params.solution_composition__init__mixing__experience_calculation)()

    experiment_name = f'ES Tuning & Experimentation {job_id} {problem}'
    experiment = Experiment(name=experiment_name,  verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_ES_GA_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    evaluation = CrossValidate(
        estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(
        n_splits=1, test_size=0.25, random_state=random_state), n_jobs=1)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()