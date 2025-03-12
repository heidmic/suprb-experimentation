import numpy as np
import click
import mlflow
from optuna import Trial

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import metrics, base
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import ElasticNet, Lasso, Ridge, LogisticRegression
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate, CustomSwapEvaluation
from experiments.mlflow import log_experiment, _log_experiment, log_run, log_run_result
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.wrapper import SupRBWrapper
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import ga
from suprb.optimizer.rule import es, origin, mutation
from suprb.solution.initialization import RandomInit
import suprb.solution.mixing_model as mixing_model
from suprb.rule.initialization import MeanInit


random_state = 42

fold_amount = 8
suprb_iter = 32
tuning_iter = 100
fold_amount = 2
suprb_iter = 2
tuning_iter = 1

Regressors = {'lasso': Lasso(alpha=0.01, random_state=random_state),
                  'elasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=random_state, fit_intercept=True,),
                   'ridge': Ridge(alpha=0.01, random_state=random_state)}
Classifiers = {'l1': LogisticRegression(penalty='l1', C=100, random_state=random_state, solver='saga'),
               'l2': LogisticRegression(penalty='l2', C=100, random_state=random_state, solver='saga'),
               'elasticnet': LogisticRegression(penalty='elasticnet', C=100, l1_ratio=0.5, random_state=random_state, solver='saga')}

def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)
    

def is_classification(name: str) -> bool:
    from problems import datasets
    return datasets.is_classification(name)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='abalone')
@click.option('-l', '--local_model', type=click.STRING, default='elasticnet')
def run(problem: str, local_model: str):
    print(f"Problem is {problem}, with local model {local_model}")
    isClassifier = is_classification(name=problem)
    X, y = load_dataset(name=problem, return_X_y=True)
    if not isClassifier:
        X, y = scale_X_y(X, y)
    elif isClassifier:
        X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    X, y = shuffle(X, y, random_state=random_state)
    
    models = Regressors
    tuning_scoring = 'neg_mean_squared_error'
    evaluation_metric = 'neg_mean_squared_error'
    mixing = mixing_model.ErrorExperienceHeuristic()
    matching_type = rule.matching.OrderedBound()
    fitness = rule.fitness.VolumeWu()
    if isClassifier:
        models = Classifiers
        tuning_scoring='accuracy'
        evaluation_metric = ['accuracy', 'f1']
        mixing = mixing_model.ErrorExperienceClassification()


    model = models[local_model]

    estimator = SupRBWrapper(
            
                rule_generation=es.ES1xLambda(
                operator='&',
                n_iter=1000,
                delay=30,
                init=rule.initialization.MeanInit(fitness=fitness,
                                                model=model, matching_type=matching_type),
                mutation=mutation.HalfnormIncrease(),
                origin_generation=origin.SquaredError(),
            ),
            solution_composition=ga.GeneticAlgorithm(n_iter=32, population_size=32, selection=ga.selection.Tournament()),
            solution_composition__init__mixing = mixing,
            n_iter=suprb_iter,
            n_rules=4,
            verbose=0,
            logger=CombinedLogger([('stdout', StdoutLogger()), ('default', DefaultLogger())]),
    )
    
    tuning_params = dict(
            estimator=estimator,
            random_state=random_state,
            cv=4,
            n_jobs_cv=4,
            n_jobs=4,
            n_calls=tuning_iter,
            timeout=60*60*24,  # 24 hours
            scoring=tuning_scoring,
            verbose=1
    )
        
    @param_space()
    def suprb_ES_GA_space(trial: Trial, params: Bunch):
        # ES
        sigma_space = [0, np.sqrt(X.shape[1])]

        params.rule_generation__mutation__sigma = trial.suggest_float('rule_generation__mutation__sigma', *sigma_space)
        #params.rule_generation__init__fitness = getattr(MeanInit, "fitness")()
        #if isinstance(params.rule_generation__init__fitness, rule.fitness.VolumeWu):
        #    params.rule_generation__init__fitness__alpha = trial.suggest_float(
        #        'rule_generation__init__fitness__alpha', 0.01, 0.2)

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
        #params.solution_composition__init__mixing = mixing
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
        
        params.rule_generation__init__model__tol = trial.suggest_float('rule_generation__init__model__tol', 1e-4, 1e-1)
        # Local_model
        if isClassifier:#isinstance(params.rule_generation__init__model, base.ClassifierMixin):
            #params.rule_generation__init__model__solver = trial.suggest_categorical('rule_generation__init__model__solver', ['saga', 'liblinear'])
            #params.rule_generation__init__model__C = trial.suggest_float('rule_generation__init__model__C', 0.01, 2)
            params.rule_generation__init__model__max_iter = trial.suggest_int('rule_generation__init__model__max_iter', 100, 1000)

    
    random_amount = fold_amount

    experiment_name = f'SupRB Tuning p:{problem}; l:{local_model}'
    print(experiment_name)
    experiment = Experiment(name=experiment_name,  verbose=0)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)
    experiment.with_tuning(suprb_ES_GA_space, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(random_amount)
    experiment.with_random_states(random_states, n_jobs=random_amount)

    evaluation = CrossValidate(
        estimator=estimator, X=X, y=y, random_state=random_state, verbose=5)

    experiment.perform(evaluation, cv=ShuffleSplit(
        n_splits=fold_amount, test_size=0.25, random_state=random_state), n_jobs=fold_amount, scoring=evaluation_metric)
    
    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)

    trained_estimators = []
    for exp in experiment.experiments:
        trained_estimators.extend(exp.estimators_)
    base_model = models[local_model]#models.pop(local_model)
    for model in models:
        swapped_name = f'{experiment_name} Swapped n:{model}'
        print(f"Swapping {model}")
        
        # Custom splitting and evaluation handled by eval
        swapped_experiment = Experiment(name=swapped_name,  verbose=0)
        swapped_experiment.with_random_states(random_states, n_jobs=random_amount)

        eval = CustomSwapEvaluation(dummy_estimator=estimator, X=X, y=y, random_state=random_state,
                                            verbose=5, local_model=models[model], trained_estimators=trained_estimators, isClass=isClassifier)
        experiment.perform(eval, cv=ShuffleSplit(
            n_splits=fold_amount, test_size=0.25, random_state=random_state), n_jobs=fold_amount)
        
        mlflow.set_experiment(swapped_name)
        print("log_experiment: " + str(log_experiment(swapped_experiment)))
        print("Results: " + str(getattr(swapped_experiment, 'results_', None)))
        print("Estimators: " + str(getattr(swapped_experiment, 'estimators_', None)))
        
        '''
        # Custom splitting and evaluation done manually
        splitter = ShuffleSplit(n_splits=fold_amount, test_size=0.25, random_state=random_states[0])
        for i, (train_index, test_index) in enumerate(splitter.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator = trained_estimators[i]
            estimator.model_swap_fit(model, X_train, y_train)
            prediction = estimator.predict(X_test)
            scorer = mean_squared_error if not isClass else accuracy_score
            score = scorer(y_test, prediction)
            name = f'{swapped_name} fold-{i}'
            swapped_experiment = Experiment(name=name,  verbose=0)
            swapped_experiment.estimators_ = [estimator]
            result_dict = {'test_score': [score]}
            swapped_experiment.results_ = result_dict
            mlflow.set_experiment(name)
            _log_experiment(swapped_experiment, parent_name=f'Swaps of {base_model}', depth=0)
        '''
if __name__ == '__main__':
    run()
