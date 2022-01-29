import mlflow
import numpy as np
from optuna import Trial
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from skopt.space import Integer, Real

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from experiments.parameter_search.skopt import SkoptTuner
from problems import scale_X_y
from problems.datasets import load_airfoil_self_noise
import xcsf
from sklearn.utils import Bunch, shuffle
import click
import optuna
from sklearn.model_selection import ShuffleSplit

random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='concrete_strength')
def run(problem: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = xcsf.XCS(X.shape[1], 1, 1)

    shared_tuning_params = dict(
        n_trials=128,
        timeout=90 * 60 * 60,  # 90 hours
    )

    # tuner = OptunaTuner(X_train=X, y_train=y, scoring='fitness',
    # **shared_tuning_params)

    # Create the base experiment, using some default tuner
    # experiment = Experiment(name='XCSF',  verbose=10)
    from sklearn.model_selection import cross_validate, KFold
    @param_space()
    def optuna_objective(trial: optuna.Trial, params: Bunch):
        params.MAX_TRIALS = trial.suggest_int('MAX_TRIALS', 10000, 1000000)
        params.POP_SIZE = trial.suggest_int('POP_SIZE', 250, 2500)
        params.P_CROSSOVER = trial.suggest_float('P_CROSSOVER', 0.5, 1)
        params.P_EXPLORE = trial.suggest_float('P_EXPLORE', 0.5, 0.9)
        params.NU = trial.suggest_int('NU', 1, 5)
        params.THETA_EA = trial.suggest_int('THETA_EA', 25, 50)
        params.EA_SUBSUMPTION = trial.suggest_categorical('EA_SUBSUMPTION',
                                                          [True, False])
        params.EA_SELECT_TYPE = trial.suggest_categorical('EA_SELECT_TYPE',
                                                          ["roulette",
                                                           "tournament"])

        scores = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring='neg_mean_squared_error',
            cv=4,
            return_estimator=True,
            verbose=10,
            fit_params=params
        )

    study = optuna.create_study()
    study.optimize(optuna_objective,**shared_tuning_params)
    print(True)
    # experiment.with_tuning(optuna_objective, tuner=tuner)
    #
    # random_states = np.random.SeedSequence(random_state).generate_state(4)
    # experiment.with_random_states(random_states, n_jobs=4)
    #
    # # Evaluation using cross-validation and an external test set
    # #evaluation = CrossValidate(estimator=estimator, X=X, y=y,
    #  #                          random_state=random_state, verbose=10)
    #
    # #experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8,
    # # test_size=0.25, random_state=random_state), n_jobs=8)
    # experiment.perform(evaluation=None)

    #mlflow.set_experiment(problem)
    #log_experiment(experiment)


if __name__ == '__main__':
    run()