from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from skopt.space import Integer, Categorical
from suprb2 import SupRB2

from experiment.base import ParameterTuning, FixedParameters, NestedExperiment, ParameterList, Evaluation

if __name__ == '__main__':
    random_state = 42

    X, y = make_regression(n_samples=1000, n_features=5, noise=5, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    experiment = ParameterTuning(
        FixedParameters(
            NestedExperiment([
                ParameterTuning(
                    ParameterList(
                        Evaluation(SupRB2(verbose=10, n_iter=1)),
                        parameter_name='estimator__n_rules',
                        parameter_list=range(2, 16, 4)),
                    parameter_space={'n_initial_rules': Integer(1, 16)},
                ),
                ParameterTuning(
                    ParameterList(
                        Evaluation(DecisionTreeRegressor()),
                        parameter_name='estimator__max_depth',
                        parameter_list=range(1, 5),
                    ),
                    parameter_space={'criterion': Categorical(['squared_error', 'absolute_error'])},
                ),
            ]),
            parameters=dict(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )
        ),
        random_state=0,
    )()

