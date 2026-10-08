import click
import mlflow
import numpy as np
from sklearn.model_selection import ShuffleSplit
from suprb.rule.matching import OrderedBound, UnorderedBound, CenterSpread, MinPercentage

from experiments import Experiment
from experiments.evaluation import CrossValidate
from experiments.mlflow import log_experiment
from problems import scale_X_y
from runs.hyperrectangle_tests.configurations.shared_config import (
    load_dataset,
    global_params,
    estimator,
    random_state,
    individual_dataset_params,
)
from runs.hyperrectangle_tests.configurations.dataset_params import params_obr, params_ubr, params_csr, params_mpr

datasets = {
    0: "parkinson_total",
    1: "protein_structure",
    2: "airfoil_self_noise",
    3: "concrete_strength",
    4: "combined_cycle_power_plant",
}

# CHANGE FOR TESTING (Choices: OBR, UBR, CSR and MPR)
representation = "MPR"

# The individual parameters for the respective Representation
representation_params = {"OBR": params_obr, "UBR": params_ubr, "CSR": params_csr, "MPR": params_mpr}
# Which representation SupRB should be set to
matching_type = {
    "OBR": OrderedBound(np.array([])),
    "UBR": UnorderedBound(np.array([])),
    "CSR": CenterSpread(np.array([])),
    "MPR": MinPercentage(np.array([])),
}


@click.command()
@click.option("-p", "--problem", type=click.STRING, default="airfoil_self_noise")
def run(problem: str):
    print(f"Problem is {problem}, Representation is {representation}")

    # Set all the representation-dependant parameters
    dataset_params = representation_params[representation]
    estimator.matching_type = matching_type[representation]

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    params = global_params | individual_dataset_params.get(problem, {}) | dataset_params.get(problem, {})

    experiment = Experiment(name=f"{representation}-{problem} Evaluation", params=params, verbose=10)

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=2)

    # Evaluation
    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)
    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)

    mlflow.set_experiment(f"{representation}_{problem}")
    log_experiment(experiment)


if __name__ == "__main__":
    run()
