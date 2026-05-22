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
from sklearn.utils import shuffle
from configurations.shared_config import load_dataset, estimator, random_state

from configurations.dataset_params import params_ga, params_saga1, params_saga2, params_saga3

# CHANGE FOR TESTING (Choices: GA, SAGA1, SAGA2 and SAGA3)
solution_composition = "SAGA1"

# The individual parameters for the respective Solution Composition
solution_composition_params = {"GA": params_ga, "SAGA1": params_saga1, "SAGA2": params_saga2, "SAGA3": params_saga3}


def run(problem: str, _random_state: int):
    print(f"Problem is {problem}, Solution Composition Optimizer is {solution_composition}")

    dataset_params = solution_composition_params[solution_composition]

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)

    if problem == "protein_structure":
        X, y = shuffle(X, y, random_state=random_state, n_samples=9146)
    else:
        X, y = shuffle(X, y, random_state=random_state)

    params = dataset_params.get(problem, {})

    experiment = Experiment(name=f"Profiling {solution_composition}_{problem}", params=params, verbose=10)

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

# Performs 8 different runs for each Solution Composition/Learning Task-combination and stores information in csv-Files
if __name__ == "__main__":
    random_states_ = np.random.SeedSequence(random_state).generate_state(8)
    for learning_task in datasets.values():
        count = 0
        directory = os.path.join(f"{solution_composition}")
        if not os.path.exists(directory):
            os.mkdir(directory)
        directory = os.path.join(f"{solution_composition}/{learning_task}")
        if not os.path.exists(directory):
            os.mkdir(directory)

        for seed in random_states_:
            pr = cProfile.Profile()
            pr.enable()
            run(problem=learning_task, _random_state=seed)
            pr.disable()
            csv = prof_to_csv(pr)
            with open(f"{solution_composition}/{learning_task}/{count}.csv", "w+") as f:
                f.write(csv)
            count += 1
