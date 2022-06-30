import click
import mlflow
import numpy as np
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from problems import scale_X_y
from shared_config import load_dataset, global_params, estimator, random_state, dataset_params, \
    individual_dataset_params

datasets = {0: 'parkinson_total', 1: 'protein_structure', 2: 'airfoil_self_noise',
            3: 'concrete_strength', 4: 'combined_cycle_power_plant'}

sigma = {0: 0.25, 1: 1.00, 2: 1.75, 3: 2.50}


def run(problem: str = 'parkinson_total', optimizer: str = 'ga', sigma_choice: int = 0):
    # my_index = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    # problem = datasets.get(my_index)
    print(f"Problem is {problem}, optimizer is {optimizer}, sigma_tuple is {sigma[sigma_choice]}")
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)

    params = global_params | individual_dataset_params.get(problem, {}) | dataset_params.get(problem, {})

    # Replace the current sigma
    params['rule_generation__mutation__sigma_lower'] = sigma[sigma_choice]
    params['rule_generation__mutation__sigma_prop'] = sigma[sigma_choice]
    experiment = Experiment(name=f'{problem}Ada Evaluation', params=params, verbose=10)

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    # Evaluation
    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(problem + "_" + str(sigma[sigma_choice]))
    log_experiment(experiment)


if __name__ == '__main__':
    for choice in range(4):
        run(sigma_choice=choice)
