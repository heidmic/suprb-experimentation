import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
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

Regressors = {'lasso': Lasso(alpha=0.01, random_state=random_state, tol=1e-3),
                  'elasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, fit_intercept=True,),
                   'ridge': Ridge(alpha=0.01, random_state=random_state)}
Classifiers = {'l1': LogisticRegression(penalty='l1', C=1.0, random_state=random_state, solver='saga'),
               'l2': LogisticRegression(penalty='l2', C=1.0, random_state=random_state, solver='saga'),
               'elasticnet': LogisticRegression(penalty='elasticnet', C=1.0, l1_ratio=0.5, random_state=random_state, solver='saga')}

def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)
    

def is_class(name: str) -> bool:
    from problems import datasets
    return datasets.is_classification(name)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-l', '--local_model', type=click.STRING, default='ridge')
def run(problem: str, local_model: str):
    print(f"Problem is {problem}, with local model {local_model}")
    isClass = is_class(name=problem)
    X, y = load_dataset(name=problem, return_X_y=True)
    if not isClass:
        X, y = scale_X_y(X, y)
    elif isClass:
        X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X) 
    X, y = shuffle(X, y, random_state=random_state)

    model = Ridge(alpha=0.01,random_state=random_state)
    models = Regressors
    scoring = 'neg_mean_squared_error'
    if isClass:
        models = Classifiers
        model = LogisticRegression(random_state=random_state)
        scoring='accuracy'
    
    if local_model == 'Lasso':
        model = Lasso(alpha=0.01, random_state=random_state, tol=1e-3)
    elif local_model == 'Ridge':
        model = Ridge(alpha=0.01, random_state=random_state)

    model = models[local_model]

    estimator = SupRB(
            rule_generation=es.ES1xLambda(
                operator='&',
                n_iter=1000,
                delay=30,
                init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(),
                                                model=model),
                mutation=mutation.HalfnormIncrease(),
                origin_generation=origin.SquaredError(),
            ),
            solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32, selection=ga.selection.Tournament()),
            n_iter=64,
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
            n_calls=200,
            timeout=60*60*24,  # 24 hours
            scoring=scoring,
            verbose=10
    )
        
    @param_space()
    def suprb_ES_GA_space(trial: Trial, params: Bunch):
        # ES
        sigma_space = [0, np.sqrt(X.shape[1])]

        params.rule_generation__mutation__sigma = trial.suggest_float('rule_generation__mutation__sigma', *sigma_space)
        params.rule_generation__init__fitness__alpha = trial.suggest_float(
            'rule_generation__init__fitness__alpha', 0.01, 0.2)

        # GA
        params.solution_composition__selection__k = trial.suggest_int('solution_composition__selection__k', 3, 10)

        params.solution_composition__crossover = trial.suggest_categorical(
            'solution_composition__crossover', ['NPoint', 'Uniform'])
        params.solution_composition__crossover = getattr(ga.crossover, params.solution_composition__crossover)()

        if isinstance(params.solution_composition__crossover, ga.crossover.NPoint):
            params.solution_composition__crossover__n = trial.suggest_int('solution_composition__crossover__n', 1, 10)

        params.solution_composition__mutation__mutation_rate = trial.suggest_float(
            'solution_composition__mutation_rate', 0, 0.1)

        # Mixing
        #params.solution_composition__init__mixing__filter_subpopulation__rule_amount = 4
        #params.solution_composition__init__mixing__experience_weight = 1.0

        params.solution_composition__init__mixing__filter_subpopulation = getattr(mixing_model, "FilterSubpopulation")()
        params.solution_composition__init__mixing__experience_calculation = getattr(
            mixing_model, "ExperienceCalculation")()

        # Upper and lower bound clip the experience into a given range
        # params.solution_composition__init__mixing__experience_calculation__lower_bound = trial.suggest_float(
        #     'solution_composition__init__mixing__experience_calculation__lower_bound', 0, 10)

        if isinstance(params.solution_composition__init__mixing__experience_calculation, mixing_model.CapExperienceWithDimensionality):
            params.solution_composition__init__mixing__experience_calculation__upper_bound = trial.suggest_float(
                'solution_composition__init__mixing__experience_calculation__upper_bound', 2, 5)
        else:
            params.solution_composition__init__mixing__experience_calculation__upper_bound = trial.suggest_int(
                'solution_composition__init__mixing__experience_calculation__upper_bound', 20, 50)

    experiment_name = f'SupRB Tuning p:{problem}; l:{local_model}'
    print(experiment_name)
    experiment = Experiment(name=experiment_name,  verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_ES_GA_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    evaluation = CrossValidate(
        estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(
        n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)
    
    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)
    
    models.pop(local_model)
    for model in models:
        trained_estimator = experiment.estimators_[0]
        new_model = model
        swapped_elitist = trained_estimator.swap_models(new_model)

        experimentSwapped = Experiment(name=f'{experiment_name} Swapped n:{new_model}', verbose=10)
        experimentSwapped.with_random_states(random_states, n_jobs=8)

        evaluation = CrossValidate(
            estimator=swapped_elitist, X=X, y=y, random_state=random_state, verbose=10)

        experimentSwapped.perform(evaluation, cv=ShuffleSplit(
            n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)
        
        mlflow.set_experiment(name=f'{experiment_name} Swapped n:{new_model}')
        log_experiment(experimentSwapped)
    print("Number of trained estimators: ", experiment.estimators_.__len__)


if __name__ == '__main__':
    run()
