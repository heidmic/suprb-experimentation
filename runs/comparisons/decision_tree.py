import mlflow
import numpy as np
import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from skopt.space import Integer, Real

from experiments import Experiment
from experiments.evaluation import CrossValidateTest
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from experiments.parameter_search.skopt import SkoptTuner
from problems import scale_X_y
from problems.datasets import load_airfoil_self_noise
import click


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-j', '--job_id', type=click.STRING, default='NA')
def run(problem: str, job_id: str):
    random_state = 42

    X, y = load_airfoil_self_noise()
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    estimator = DecisionTreeRegressor(random_state=random_state)

    default_tuner = SkoptTuner(
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        n_calls=1000,
        verbose=10,
        n_jobs_cv=4,
    )

    optuna_tuner = OptunaTuner(
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        n_calls=1000,
        verbose=10,
        n_jobs_cv=4,
    )

    # Create the base experiment, using some default tuner
    experiment_name = f'Decision Tree {job_id} {problem}'
    experiment = Experiment(name=experiment_name,   tuner=default_tuner, verbose=10)

    # Add global tuning of the `n_estimators` parameter using optuna.
    # It is tuned by itself first, and afterwards, the fixed value is propagated to nested experiments,
    # because `propagate` is not set. Note that optuna does not support merging parameter spaces.

    @param_space()
    def optuna_objective(trial: optuna.Trial, params: Bunch):

        params.criterion = trial.suggest_categorical(
            'criterion', ["squared_error", "friedman_mse", "absolute_error"])

        params.splitter = trial.suggest_categorical(
            'splitter', ["best", "random"])

        # params.min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        # params.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        # params.min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
        params.max_features = trial.suggest_categorical('max_features', ["auto", "sqrt", "log2"])
        # params.min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        # params.ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.02)

    experiment.with_tuning(optuna_objective, tuner=optuna_tuner)

    # Create a nested experiment using the MAE.
    mae_experiment = experiment.with_params({'criterion': 'absolute_error'})
    mae_experiment.name = 'MAE'

    # Tune only this experiment on some parameter
    mae_experiment.with_tuning({'min_samples_split': Real(0, 1)}, tuner=default_tuner)

    # Create a nested experiment using the MSE.
    mse_experiment = experiment.with_params({'criterion': 'squared_error'})
    mse_experiment.name = 'MSE'

    # Add global tuning of the `max_depth` parameter. Because `propagate` is set here,
    # the value is tuned new for every nested experiment.
    experiment.with_tuning({'max_depth': Integer(1, 5)}, tuner=default_tuner, propagate=True)

    random_states = np.random.SeedSequence(random_state).generate_state(4)
    experiment.with_random_states(random_states, n_jobs=4)

    # Evaluation using cross-validation and an external test set
    evaluation = CrossValidateTest(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=4)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
