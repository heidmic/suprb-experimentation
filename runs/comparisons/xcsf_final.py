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
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state  # type: ignore
from sklearn.utils.validation import check_is_fitted, check_X_y, check_array
from sklearn.metrics import mean_squared_error
# type: ignore
from sklearn.model_selection import cross_validate


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def set_xcs_params(xcs, params):
    xcs.OMP_NUM_THREADS = params["OMP_NUM_THREADS"]
    xcs.POP_INIT = params["POP_INIT"]
    xcs.POP_SIZE = params["POP_SIZE"]
    xcs.MAX_TRIALS = params["MAX_TRIALS"]
    xcs.PERF_TRIALS = params["PERF_TRIALS"]
    xcs.LOSS_FUNC = params["LOSS_FUNC"]
    xcs.HUBER_DELTA = params["HUBER_DELTA"]
    xcs.E0 = params["E0"]
    xcs.ALPHA = params["ALPHA"]
    xcs.NU = params["NU"]
    xcs.BETA = params["BETA"]
    xcs.DELTA = params["DELTA"]
    xcs.THETA_DEL = params["THETA_DEL"]
    xcs.INIT_FITNESS = params["INIT_FITNESS"]
    xcs.INIT_ERROR = params["INIT_ERROR"]
    xcs.M_PROBATION = params["M_PROBATION"]
    xcs.STATEFUL = params["STATEFUL"]
    xcs.SET_SUBSUMPTION = params["SET_SUBSUMPTION"]
    xcs.THETA_SUB = params["THETA_SUB"]
    xcs.COMPACTION = params["COMPACTION"]
    xcs.TELETRANSPORTATION = params["TELETRANSPORTATION"]
    xcs.GAMMA = params["GAMMA"]
    xcs.P_EXPLORE = params["P_EXPLORE"]
    xcs.EA_SELECT_TYPE = params["EA_SELECT_TYPE"]
    xcs.EA_SELECT_SIZE = params["EA_SELECT_SIZE"]
    xcs.THETA_EA = params["THETA_EA"]
    xcs.LAMBDA = params["LAMBDA"]
    xcs.P_CROSSOVER = params["P_CROSSOVER"]
    xcs.ERR_REDUC = params["ERR_REDUC"]
    xcs.FIT_REDUC = params["FIT_REDUC"]
    xcs.EA_SUBSUMPTION = params["EA_SUBSUMPTION"]
    xcs.EA_PRED_RESET = params["EA_PRED_RESET"]


def get_xcs_params(xcs):
    return {
        "OMP_NUM_THREADS": xcs.OMP_NUM_THREADS,
        "POP_INIT": xcs.POP_INIT,
        "POP_SIZE": xcs.POP_SIZE,
        "MAX_TRIALS": xcs.MAX_TRIALS,
        "PERF_TRIALS": xcs.PERF_TRIALS,
        "LOSS_FUNC": xcs.LOSS_FUNC,
        "HUBER_DELTA": xcs.HUBER_DELTA,
        "E0": xcs.E0,
        "ALPHA": xcs.ALPHA,
        "NU": xcs.NU,
        "BETA": xcs.BETA,
        "DELTA": xcs.DELTA,
        "THETA_DEL": xcs.THETA_DEL,
        "INIT_FITNESS": xcs.INIT_FITNESS,
        "INIT_ERROR": xcs.INIT_ERROR,
        "M_PROBATION": xcs.M_PROBATION,
        "STATEFUL": xcs.STATEFUL,
        "SET_SUBSUMPTION": xcs.SET_SUBSUMPTION,
        "THETA_SUB": xcs.THETA_SUB,
        "COMPACTION": xcs.COMPACTION,
        "TELETRANSPORTATION": xcs.TELETRANSPORTATION,
        "GAMMA": xcs.GAMMA,
        "P_EXPLORE": xcs.P_EXPLORE,
        "EA_SELECT_TYPE": xcs.EA_SELECT_TYPE,
        "EA_SELECT_SIZE": xcs.EA_SELECT_SIZE,
        "THETA_EA": xcs.THETA_EA,
        "LAMBDA": xcs.LAMBDA,
        "P_CROSSOVER": xcs.P_CROSSOVER,
        "ERR_REDUC": xcs.ERR_REDUC,
        "FIT_REDUC": xcs.FIT_REDUC,
        "EA_SUBSUMPTION": xcs.EA_SUBSUMPTION,
        "EA_PRED_RESET": xcs.EA_PRED_RESET,
    }


def default_xcs_params():
    xcs = xcsf.XCS(1, 1, 1)
    return get_xcs_params(xcs)


class XCSF(BaseEstimator, RegressorMixin):
    """
    Almost a correct sklearn wrapper for ``xcsf.XCS``. For example, it can't yet
    be pickled and some parameters are missing
    """

    def __init__(self, random_state, MAX_TRIALS=1000, POP_SIZE=200, NU=5,
                 P_CROSSOVER=0.8, P_EXPLORE=0.9, THETA_EA=50,
                 EA_SUBSUMPTION=False, EA_SELECT_TYPE="tournament"):
        self.random_state = random_state

        self.MAX_TRIALS = MAX_TRIALS
        self.POP_SIZE = POP_SIZE
        self.NU = NU
        self.P_CROSSOVER = P_CROSSOVER
        self.P_EXPLORE = P_EXPLORE
        self.THETA_EA = THETA_EA
        self.EA_SUBSUMPTION = EA_SUBSUMPTION
        self.EA_SELECT_TYPE = EA_SELECT_TYPE

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # This is required so that XCS does not (silently!?) segfault (see
        # https://github.com/rpreen/xcsf/issues/17 ).
        y = y.reshape((len(X), -1))

        random_state = check_random_state(self.random_state)

        xcs = xcsf.XCS(X.shape[1], 1, 1)  # only 1 (dummy) action
        xcs.seed(random_state.randint(np.iinfo(np.int32).max))

        params = default_xcs_params()
        configurables = {"MAX_TRIALS": self.MAX_TRIALS,
                         "POP_SIZE": self.POP_SIZE,
                         "NU": self.NU,
                         "P_CROSSOVER": self.P_CROSSOVER,
                         "P_EXPLORE": self.P_EXPLORE,
                         "THETA_EA": self.THETA_EA,
                         "EA_SUBSUMPTION": self.EA_SUBSUMPTION,
                         "EA_SELECT_TYPE": self.EA_SELECT_TYPE}
        params.update(configurables)
        set_xcs_params(xcs, params)
        print(f"fitting with {configurables}")

        xcs.action("integer")  # (dummy) integer actions

        args = {
            "min": -1,  # minimum value of a lower bound
            "max": 1,  # maximum value of an upper bound
            "spread_min": 0.1,  # minimum initial spread
            "eta":
                0,  # disable gradient descent of centers towards matched input mean
        }
        xcs.condition("hyperrectangle_csr", args)

        args = {
            "x0": 1,  # bias attribute
            "scale_factor":
                1000,  # initial diagonal values of the gain-matrix
            "lambda": 1,  # forget rate (small values may be unstable)
        }
        prediction_string = "rls_linear"
        xcs.prediction(prediction_string, args)

        xcs.fit(X, y, True)

        self.xcs_ = xcs

        return self

    def predict(self, X):
        check_is_fitted(self)

        X = check_array(X)
        self.xcs_.print_pset(True, True, True)

        return self.xcs_.predict(X)

    # def population(self):
    #     check_is_fitted(self)
    #
    #     out = io.BytesIO()
    #     with utils.stdout_redirector(out):
    #         self.xcs_.print_pset(True, True, True)
    #
    #     pop = out.getvalue().decode("utf-8")
    #     return pop


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='concrete_strength')
@click.option('-j', '--job_id', type=click.STRING, default='NA')
def run(problem: str, job_id: str):
    print(f"Problem is {problem}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    params = {}

    if problem == 'concrete_strength':
        params = {'MAX_TRIALS': 121346,
                  'POP_SIZE': 622,
                  'P_CROSSOVER': 0.8772756806442054,
                  'P_EXPLORE': 0.8313044409261099,
                  'NU': 3,
                  'THETA_EA': 39,
                  'EA_SUBSUMPTION': True,
                  'EA_SELECT_TYPE': 'roulette'}
    elif problem == 'combined_cycle_power_plant':
        params = {'MAX_TRIALS': 472201,
                  'POP_SIZE': 2495,
                  'P_CROSSOVER': 0.5405489782155791,
                  'P_EXPLORE': 0.8673932144700683,
                  'NU': 5,
                  'THETA_EA': 50,
                  'EA_SUBSUMPTION': True,
                  'EA_SELECT_TYPE': 'tournament'}
    elif problem == 'airfoil_self_noise':
        params = {'MAX_TRIALS': 374418,
                  'POP_SIZE': 1044,
                  'P_CROSSOVER': 0.9349035679222683,
                  'P_EXPLORE': 0.658032571723789,
                  'NU': 1,
                  'THETA_EA': 41,
                  'EA_SUBSUMPTION': False,
                  'EA_SELECT_TYPE': 'tournament'}
    elif problem == 'energy_cool':
        params = {'MAX_TRIALS': 373259,
                  'POP_SIZE': 1136,
                  'P_CROSSOVER': 0.7503158535428448,
                  'P_EXPLORE': 0.783389757163828,
                  'NU': 1,
                  'THETA_EA': 29,
                  'EA_SUBSUMPTION': False,
                  'EA_SELECT_TYPE': 'tournament'}

    estimator = XCSF(random_state, **params)
    experiment_name = f'XCSF Tuning {job_id} {problem}'

    # Create the base experiment, using some default tuner
    experiment = Experiment(name=experiment_name,  verbose=10)

    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=4)

    # Evaluation using cross-validation and an external test set
    evaluation = CrossValidate(estimator=estimator, X=X, y=y,
                               random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(f"XCSF_eval_{problem}")
    log_experiment(experiment)


if __name__ == '__main__':
    run()