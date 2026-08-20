import mlflow
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from suprb import SupRB
from suprb.logging.default import DefaultLogger

from problems import scale_X_y

if __name__ == "__main__":
    random_state = 42

    X, y = make_regression(n_samples=100, n_features=2, noise=5, random_state=random_state)
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model = SupRB(n_iter=4, logger=DefaultLogger())

    model.fit(X_train, y_train)

    mlflow.set_experiment("Make Regression")
    with mlflow.start_run(run_name="Single Fit"):
        logger: DefaultLogger = model.logger_

        # Log model parameters
        mlflow.log_params(logger.params_)

        # Log fitting metrics
        for key, values in logger.metrics_.items():
            for step, value in values.items():
                mlflow.log_metric(key=key, value=value, step=step)

        # Log test metrics
        mlflow.log_metric("test_score", model.score(X_test, y_test))
