from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from skopt.space import Integer, Categorical

from experiment import Experiment
from experiment.evaluation import CrossValidateTest
from experiment.parameter_search.skopt import SkoptTuner
from problems import scale_X_y
from problems.datasets import load_concrete_strength

if __name__ == '__main__':
    random_state = 42

    X, y = load_concrete_strength()
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    estimator = RandomForestRegressor(random_state=random_state)

    default_tuner = SkoptTuner(
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        n_calls=20,
        verbose=10,
        n_jobs_cv=4,
    )

    extensive_tuner = SkoptTuner(
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
        n_calls=30,
        verbose=10,
        n_jobs_cv=4,
    )

    experiment = Experiment(tuner=default_tuner, verbose=10)
    experiment.with_tuning({'n_estimators': Integer(1, 200)})

    mae_experiment = experiment.with_params({'criterion': 'absolute_error'})

    mae_experiment.with_tuning({'bootstrap': Categorical([True, False])}, tuner=extensive_tuner)

    mse_experiment = experiment.with_params({'criterion': 'squared_error'})

    experiment.with_tuning({'max_depth': Integer(1, 5)}, propagate=True)

    # Evaluation
    evaluation = CrossValidateTest(estimator=estimator, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                   random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=8, n_jobs=4)
