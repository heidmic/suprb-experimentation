import sys
import os
import csv
import time

import numpy as np
import pandas as pd

import click
import mlflow
from optuna import Trial

from sklearn.linear_model import Ridge
from sklearn.utils import Bunch, shuffle
from sklearn.model_selection import ShuffleSplit

from experiments import Experiment
from experiments.evaluation import CrossValidate, MOOCrossValidate
from experiments.mlflow import log_experiment
from experiments.parameter_search import param_space
from experiments.parameter_search.optuna import OptunaTuner
from problems import scale_X_y

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.multi_objective import MOLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import nsga2, nsga3, spea2
from suprb.optimizer.rule import es, origin, mutation, ns
from suprb.optimizer.rule import es, origin, mutation, ns
from suprb.optimizer.rule.ns.novelty_calculation import NoveltyCalculation
from suprb.optimizer.rule.ns.novelty_search_type import MinimalCriteria
from suprb.rule.subsumption import PreferSmallerVolume, PreferLargerVolume
from suprb.solution.initialization import RandomInit
from suprb.rule.matching import OrderedBound, UnorderedBound, CenterSpread, MinPercentage
import suprb.solution.mixing_model as mixing_model

opt_dict = {
    "nsga2": nsga2.NonDominatedSortingGeneticAlgorithm2,
    "nsga3": nsga3.NonDominatedSortingGeneticAlgorithm3,
    "spea2": spea2.StrengthParetoEvolutionaryAlgorithm2,
}


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets

    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


def get_n_iterations(model: SupRB):
    """ """
    for attr in ("n_iter_", "step_", "iter_", "n_iterations_", "generation_"):
        if hasattr(model, attr):
            return getattr(model, attr)
    return None


def get_hypervolume(model: SupRB):
    solution_composition = getattr(model, "solution_composition_", None)
    if solution_composition is not None and hasattr(solution_composition, "hypervolume"):
        return solution_composition.hypervolume()
    return None


def run_single_cycle(problem: str, seed: int, optimizer: str) -> tuple[SupRB, np.ndarray, np.ndarray]:
    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=seed)

    model = SupRB(
        rule_discovery=ns.NoveltySearch(
            n_iter=25,
            novelty_calculation=NoveltyCalculation(
                k_neighbor=15,
                novelty_search_type=MinimalCriteria(min_examples_matched=5),
            ),
            init=rule.initialization.HalfnormInit(fitness=rule.fitness.VolumeWu(), model=Ridge(alpha=0.01, random_state=seed)),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),
            # subsumption=PreferSmallerVolume(tolerance=0.02),
        ),
        solution_composition=opt_dict[optimizer](n_iter=32, population_size=32),
        n_iter=32,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", MOLogger())]),
        random_state=seed,
        early_stopping_patience=5,
        early_stopping_delta=0.0015,
        extra_rules_patience=1,
        extra_rules_delta=0.1000,
    )

    model.fit(X, y)
    return model, X, y


def run_multi_seed(
    problem: str = "airfoil_self_noise",
    optimizer: str = "spea2",
    n_runs: int = 25,
    base_seed: int = 0,
    out_path: str = "output/multi_seed_results.csv",
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fieldnames = [
        "seed",
        "n_iterations",
        "final_pool_size",
        "training_score",
        "hypervolume",
        "elapsed_seconds",
        "subsumed_ruels",
        "subsumed_procent",
    ]

    file_exists = os.path.isfile(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i in range(n_runs):
            seed = base_seed + i
            print(f"\n=== Run {i + 1}/{n_runs} (seed={seed}) ===")

            start = time.perf_counter()
            model, X, y = run_single_cycle(problem, seed, optimizer)
            elapsed = time.perf_counter() - start

            if i == 0:
                print("[diagnostic] model attributes:", [a for a in dir(model) if a.endswith("_") and not a.startswith("_")])
                sc = getattr(model, "solution_composition_", None)
                if sc is not None:
                    print("[diagnostic] solution_composition_ has hypervolume:", hasattr(sc, "hypervolume"))

            n_iterations = get_n_iterations(model) + 1
            final_pool_size = len(model.pool_)
            training_score = model.score(X, y)
            hypervolume = get_hypervolume(model)
            subsumed_rules = n_iterations * model.n_rules - final_pool_size
            subsumed_procent = (n_iterations * model.n_rules - final_pool_size) / (n_iterations * model.n_rules)

            row = {
                "seed": seed,
                "n_iterations": n_iterations,
                "final_pool_size": final_pool_size,
                "training_score": training_score,
                "hypervolume": hypervolume,
                "elapsed_seconds": round(elapsed, 2),
                "subsumed_ruels": subsumed_rules,
                "subsumed_procent": subsumed_procent,
            }
            writer.writerow(row)
            f.flush()

            print(row)

    print(f"\nAll {n_runs} runs complete. Results written to {out_path}")


if __name__ == "__main__":
    run_multi_seed(
        problem="concrete_strength",  # parkinson_total protein_structure airfoil_self_noise concrete_strength combined_cycle_power_plant
        optimizer="spea2",
        n_runs=25,
        base_seed=0,
        out_path="output/early5_delta0015_extra1_delta1000_200_4_ccs.csv",
    )
