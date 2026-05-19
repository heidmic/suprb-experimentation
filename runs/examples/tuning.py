import mlflow
import numpy as np
import suprb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from skopt.space import Real
from skopt.utils import point_asdict
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.rule.mutation import Normal


from experiments.parameter_search.skopt import SkoptTuner
from problems import scale_X_y

if __name__ == "__main__":

    np.seterr("raise")

    random_state = 42

    X, y = make_regression(n_samples=1000, n_features=10, noise=5, random_state=random_state)
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model = suprb.SupRB(
        n_iter=4,
        n_jobs=2,
        rule_discovery=ES1xLambda(
            init=suprb.rule.initialization.HalfnormInit(),
            mutation=Normal(),
        ),
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
    )

    param_space = {
        "rule_discovery__init__sigma": Real(0.01, 2),
        "rule_discovery__mutation__sigma": Real(0.01, 2),
    }

    tuner = SkoptTuner(
        model, X_train, y_train, scoring="r2", n_calls=10, cv=2, n_jobs_cv=2, verbose=10, random_state=random_state
    )
    tuned_params, _ = tuner(parameter_space=param_space, local_params={})

    model.set_params(**tuned_params)
    model.fit(X_train, y_train)

    mlflow.set_experiment("Make Regression")
    with mlflow.start_run(run_name="single+tuning"):
        with mlflow.start_run(run_name="tuning", nested=True):

            # Log params
            for attribute in ["tuner", "n_calls", "scoring", "cv", "parameter_space", "random_state"]:
                mlflow.log_param(attribute, getattr(tuner, attribute))

            # Log history
            for step, objective_value in enumerate(tuner.tuning_result_.func_vals):
                mlflow.log_metric("objective_function", objective_value, step=step)

            for step, params_list in enumerate(tuner.tuning_result_.x_iters):
                params = point_asdict(tuner.parameter_space, params_list)
                mlflow.log_metrics(params, step=step)

            # Log final result
            for param_name, final_value in tuned_params.items():
                mlflow.log_metric(f"{param_name}_final", final_value)

        with mlflow.start_run(run_name="single", nested=True):
            logger: DefaultLogger = model.logger_.loggers_[1][1]

            # Log model parameters
            mlflow.log_params(logger.params_)

            # Log fitting metrics
            for key, values in logger.metrics_.items():
                for step, value in values.items():
                    mlflow.log_metric(key=key, value=value, step=step)

            # Log test metrics
            mlflow.log_metric("test_score", model.score(X_test, y_test))
