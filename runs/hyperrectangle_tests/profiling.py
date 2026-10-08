import cProfile
import os
import pstats
import io
import numpy as np
from sklearn.model_selection import ShuffleSplit
from suprb.rule.matching import OrderedBound

from experiments import Experiment
from experiments.evaluation import CrossValidate
from problems import scale_X_y
from runs.hyperrectangle_tests.configurations.shared_config import (
    load_dataset,
    global_params,
    estimator,
    random_state,
    individual_dataset_params,
)

from runs.hyperrectangle_tests.configurations.dataset_params import params_obr, params_ubr, params_csr, params_mpr

# CHANGE FOR TESTING (Choices: OBR, UBR, CSR and MPR)
representation = "OBR"

# The individual parameters for the respective Representation
representation_params = {"OBR": params_obr, "UBR": params_ubr, "CSR": params_csr, "MPR": params_mpr}
# Which representation SupRB should be set to TODO
matching_type = {
    "OBR": OrderedBound(np.array([])),
    "UBR": OrderedBound(np.array([])),
    "CSR": OrderedBound(np.array([])),
    "MPR": OrderedBound(np.array([])),
}


def run(problem: str, _random_state: int):
    print(f"Problem is {problem}, Representation is {representation}")

    dataset_params = representation_params[representation]
    estimator.matching_type = matching_type[representation]

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)

    params = global_params | individual_dataset_params.get(problem, {}) | dataset_params.get(problem, {})

    experiment = Experiment(name=f"Profiling {representation}_{problem}", params=params, verbose=10)

    # Repeat evaluations with several random states
    random_states = np.random.SeedSequence(_random_state).generate_state(1)
    experiment.with_random_states(random_states, n_jobs=1)

    # Evaluation
    evaluation = CrossValidate(estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state), n_jobs=1)


def prof_to_csv(prof: cProfile.Profile):
    out_stream = io.StringIO()
    pstats.Stats(prof, stream=out_stream).sort_stats("time").print_stats()
    result = out_stream.getvalue()
    # chop off header lines
    result = "ncalls" + result.split("ncalls")[-1]
    lines = [",".join(line.rstrip().split(None, 5)) for line in result.split("\n")]
    return "\n".join(lines)


datasets = {
    0: "parkinson_total",
    1: "protein_structure",
    2: "airfoil_self_noise",
    3: "concrete_strength",
    4: "combined_cycle_power_plant",
}

# Performs 10 different runs for each Representation/Learning Task-combination and stores information in csv-Files
if __name__ == "__main__":
    random_states_ = np.random.SeedSequence(random_state).generate_state(10)
    for learning_task in datasets.values():
        count = 0
        directory = os.path.join(f"{learning_task}")
        if not os.path.exists(directory):
            os.mkdir(directory)
        for seed in random_states_:
            pr = cProfile.Profile()
            pr.enable()
            run(problem=learning_task, _random_state=seed)
            pr.disable()
            csv = prof_to_csv(pr)
            with open(f"{representation}/{learning_task}/{count}.csv", "w+") as f:
                f.write(csv)
            count += 1
