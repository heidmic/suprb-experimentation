import click
import mlflow
import numpy as np
from sklearn.model_selection import ShuffleSplit
from suprb.rule.matching import GaussianKernelFunction, GaussianKernelFunctionGeneralEllipsoids, OrderedBound

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from problems import scale_X_y
from runs.hyperellipsoid_tests.configurations.shared_config import load_dataset, global_params, estimator, \
    random_state, individual_dataset_params
from runs.hyperellipsoid_tests.configurations.dataset_params import params_ordered_bound, params_ellipsoid, params_general_ellipsoid

datasets = {0: 'airfoil_self_noise'}

# CHANGE FOR TESTING (Choices: OBR, UBR, CSR and MPR)
representation = 'OBR'

# The individual parameters for the respective Representation
representation_params = {'OBR': params_ordered_bound, 'HE': params_ellipsoid, 'GHE': params_general_ellipsoid}
# Which representation SupRB should be set to
matching_type = {'OBR': OrderedBound(np.array([])), 'HE': GaussianKernelFunction(np.array([]), np.array([])),
                 'GHE': GaussianKernelFunctionGeneralEllipsoids(np.array([]), np.array([]))}


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
def run(problem: str):
    print(f"Problem is {problem}, Representation is {representation}")
    test = estimator.get_params().keys()
    # Set all the representation-dependant parameters
    dataset_params = representation_params[representation]
    estimator.matching_type = matching_type[representation]

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    params = global_params | individual_dataset_params.get(problem, {}) | dataset_params.get(problem, {})

    experiment = Experiment(name=f'{representation}-{problem} Evaluation', params=params, verbose=10)

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    # Evaluation
    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)
    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(f'{representation}_{problem}')
    log_experiment(experiment)


if __name__ == '__main__':
    run()