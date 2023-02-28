import mlflow
import numpy as np
import optuna
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch, shuffle
from skopt.space import Integer, Real
from sklearn.model_selection import ShuffleSplit


from experiments import Experiment
from experiments.evaluation import CrossValidate
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
    X, y = shuffle(X, y, random_state=random_state)

    estimator = DecisionTreeRegressor(random_state=random_state)

    tuning_params = dict(
        estimator=estimator,
        random_state=random_state,
        cv=4,
        n_jobs_cv=4,
        n_jobs=4,
        n_calls=1000,
        timeout=72 * 60 * 60,  # 24 hours
        verbose=10
    )

    # Add global tuning of the `n_estimators` parameter using optuna.
    # It is tuned by itself first, and afterwards, the fixed value is propagated to nested experiments,
    # because `propagate` is not set. Note that optuna does not support merging parameter spaces.

    @param_space()
    def optuna_objective(trial: optuna.Trial, params: Bunch):

        params.criterion = trial.suggest_categorical(
            'criterion', ["squared_error", "friedman_mse", "absolute_error"])

        params.splitter = trial.suggest_categorical(
            'splitter', ["best", "random"])

        params.min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        params.min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
        params.min_weight_fraction_leaf = trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.5)
        params.max_features = trial.suggest_categorical('max_features', ["auto", "sqrt", "log2"])
        params.min_impurity_decrease = trial.suggest_float('min_impurity_decrease', 0.0, 0.5)
        params.ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.02)

    experiment_name = f'Decision Tree {job_id} {problem}'
    experiment = Experiment(name=experiment_name, verbose=10)

    tuner = OptunaTuner(X_train=X, y_train=y, **tuning_params)

    # Add global tuning of the `max_depth` parameter. Because `propagate` is set here,
    # the value is tuned new for every nested experiment.
    experiment.with_tuning(optuna_objective, tuner=tuner)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(experiment_name)
    log_experiment(experiment)


if __name__ == '__main__':
    run()
