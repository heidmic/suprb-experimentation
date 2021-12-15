import suprb2
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skopt.space import Real
from suprb2 import SupRB2
from suprb2.logging.stdout import StdoutLogger
from suprb2.optimizer.rule.es import ES1xLambda

from experiment.parameter_search.skopt import SkoptTuner

if __name__ == '__main__':
    random_state = 42

    X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=random_state)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = StandardScaler().fit_transform(y.reshape((-1, 1))).reshape((-1,))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model = suprb2.SupRB2(
        n_iter=4,
        rule_generation=ES1xLambda(
            init=suprb2.rule.initialization.HalfnormInit(),
            mutation=suprb2.optimizer.rule.es.mutation.Normal(),
        ),
        logger=StdoutLogger(),
    )

    param_space = {
        'rule_generation__init__sigma': Real(0.01, 2),
        'rule_generation__mutation__sigma': Real(0.01, 2),
    }

    tuner = SkoptTuner(model, X_train, y_train, scoring='r2', parameter_space=param_space, n_calls=10, cv=2, n_jobs_cv=2,
                       verbose=10, random_state=random_state)
    tuned_params = tuner.tune()
    print("tuned params", tuned_params)

    model.set_params(**tuned_params)
    model.fit(X_train, y_train)

    print("test r2 score:", model.score(X_test, y_test))
