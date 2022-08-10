import click
import mlflow
import numpy as np
from sklearn.model_selection import ShuffleSplit
from suprb.rule.matching import OrderedBound

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from problems import scale_X_y
from runs.hyperrectangle_tests.configurations.shared_config import load_dataset, global_params, estimator, \
    random_state, individual_dataset_params
from runs.hyperrectangle_tests.configurations.dataset_params import params_obr, params_ubr, params_csr, params_mpr

datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}

# CHANGE FOR TESTING (Choices: OBR, UBR, CSR and MPR)
representation = 'OBR'

# The individual parameters for the respective Representation
representation_params = {'OBR': params_obr, 'UBR': params_ubr, 'CSR': params_csr, 'MPR': params_mpr}
# Which representation SupRB should be set to TODO
matching_type = {'OBR': OrderedBound(np.array([])), 'UBR': OrderedBound(np.array([])),
                 'CSR': OrderedBound(np.array([])), 'MPR': OrderedBound(np.array([]))}

sigma_obr = {0: 0.25, 1: 1.00, 2: 1.75, 3: 2.50}
sigma_ubr = {0: 0.25, 1: 1.00, 2: 1.75, 3: 2.50}
sigma_csr = {0: [0.02, 0.25], 1: [0.02, 1.00], 2: [0.02, 1.75], 3: [0.02, 2.50]}
sigma_mpr = {0: [0.25, 0.25], 1: [1.00, 1.00], 2: [1.75, 1.75], 3: [2.50, 2.50]}

sigma_representations = {'OBR': sigma_obr, 'UBR': sigma_ubr, 'CSR': sigma_csr, 'MPR': sigma_mpr}


def run(problem: str = 'parkinson_total', sigma_choice: int = 0):
    # my_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # problem = datasets.get(my_index)
    print(f"Problem is {problem}, Representation is {representation},"
          f" sigma is {sigma_representations[representation][sigma_choice]}")
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)

    # Set all the representation-dependant parameters
    dataset_params = representation_params[representation]
    estimator.matching_type = matching_type[representation]
    sigma = sigma_representations[representation]

    params = global_params | individual_dataset_params.get(problem, {}) | dataset_params.get(problem, {})

    # Replace the current sigma
    params['rule_generation__mutation__sigma'] = sigma[sigma_choice]
    experiment = Experiment(name=f'{problem}Ada Evaluation', params=params, verbose=10)

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    # Evaluation
    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(f'{representation}_{problem}_{sigma[sigma_choice]}')
    log_experiment(experiment)


if __name__ == '__main__':
    for choice in range(4):
        run(sigma_choice=choice)
