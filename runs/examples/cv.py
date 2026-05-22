from sklearn.model_selection import train_test_split
from suprb import suprb, rule
from suprb.logging.combination import CombinedLogger
from suprb.logging.default import DefaultLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.rule import es
from suprb.optimizer.rule.mutation import Normal


from experiments import Experiment
from experiments.evaluation import CrossValidateTest
from problems import scale_X_y
from problems.datasets import load_gas_turbine

if __name__ == "__main__":
    random_state = 42

    X, y = load_gas_turbine()
    X, y = scale_X_y(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    estimator = suprb.SupRB(
        n_iter=4,
        n_jobs=2,
        rule_discovery=es.ES1xLambda(
            init=rule.initialization.HalfnormInit(sigma=0.8),
            mutation=Normal(sigma=0.4),
        ),
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", DefaultLogger())]),
    )

    experiment = Experiment(verbose=10)
    evaluation = CrossValidateTest(
        estimator=estimator,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=random_state,
        verbose=10,
    )
    experiment.perform(evaluation, cv=8, n_jobs=4)
