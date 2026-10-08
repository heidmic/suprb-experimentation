import sys

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
import time

from suprb import rule, SupRB
from suprb.logging.combination import CombinedLogger
from suprb.logging.multi_objective import MOLogger
from suprb.logging.stdout import StdoutLogger
from suprb.optimizer.solution import nsga2, nsga3, spea2
from suprb.optimizer.rule import es, origin, mutation, ns
from suprb.rule.subsumption import PreferSmallerVolume, PreferLargerVolume
from suprb.solution.initialization import RandomInit
from suprb.rule.matching import OrderedBound, UnorderedBound, CenterSpread, MinPercentage
import suprb.solution.mixing_model as mixing_model

from problems import scale_X_y

random_state = 44

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


def get_effective_bounds(match) -> np.ndarray:
    if isinstance(match, OrderedBound):
        return match.bounds

    if isinstance(match, UnorderedBound):
        lower = np.min(match.bounds, axis=1)
        upper = np.max(match.bounds, axis=1)
        return np.stack([lower, upper], axis=1)

    if isinstance(match, CenterSpread):
        lower = match.bounds[:, 0] - match.bounds[:, 1]
        upper = match.bounds[:, 0] + match.bounds[:, 1]
        return np.stack([lower, upper], axis=1)

    if isinstance(match, MinPercentage):
        lower = match.bounds[:, 0]
        upper = lower + match.bounds[:, 1] * (1 - lower)
        return np.stack([lower, upper], axis=1)


def bounds_contains(outer_bounds: np.ndarray, inner_bounds: np.ndarray) -> bool:
    return bool(np.all(outer_bounds[:, 0] <= inner_bounds[:, 0]) and np.all(outer_bounds[:, 1] >= inner_bounds[:, 1]))


def run_single_cycle(problem: str, job_id: str, optimizer: str) -> SupRB:

    print(f"Problem is {problem}, with job id {job_id} and optimizer {optimizer}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    model = SupRB(
        rule_discovery=es.ES1xLambda(
            operator="&",
            n_iter=1000,
            delay=30,
            init=rule.initialization.MeanInit(fitness=rule.fitness.VolumeWu(), model=Ridge(alpha=0.01, random_state=random_state)),
            mutation=mutation.HalfnormIncrease(),
            origin_generation=origin.SquaredError(),
            subsumption=PreferSmallerVolume(tolerance=0.0),
        ),
        solution_composition=opt_dict[optimizer](n_iter=32, population_size=32),
        n_iter=200,
        n_rules=4,
        verbose=10,
        logger=CombinedLogger([("stdout", StdoutLogger()), ("default", MOLogger())]),
        random_state=random_state,
        early_stopping_patience=20,
    )

    start = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - start
    print(f"Training took {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
    print([r.numerosity_ for r in model.pool_])
    print(model.rule_discovery_.subsumption)
    print(len(model.pool_))
    print(model.elitist_.genome.shape if hasattr(model.elitist_, "genome") else "check attribute name")
    return model


def get_final_pool(model: SupRB) -> list:

    return model.pool_


def pred_diff_on_overlap(containing_rule, contained_rule) -> np.ndarray:
    mask_i = containing_rule.match_set_
    mask_j = contained_rule.match_set_

    # cumulative count of True's up to and including each position in mask_i;
    # subtracting 1 gives the index into the compressed pred_i array
    cum = np.cumsum(mask_i) - 1
    idx_in_i = cum[mask_j]  # valid because mask_j ⊆ mask_i

    pred_i_on_overlap = containing_rule.pred_[idx_in_i]
    return pred_i_on_overlap - contained_rule.pred_


def analyze_pool(pool: list, tolerance: float = 0.0) -> pd.DataFrame:
    n = len(pool)
    bounds = [get_effective_bounds(r.match) for r in pool]
    match_sets = [r.match_set_ for r in pool]
    errors = [r.error_ for r in pool]

    records = []
    for i in range(n):
        bounds_i = bounds[i]
        match_i = match_sets[i]
        for j in range(n):
            if i == j:
                continue
            bounds_j = bounds[j]
            match_j = match_sets[j]

            if match_j.sum() == 0:
                continue  # degenerate rule, skip

            contains = bounds_contains(bounds_i, bounds_j)
            if not contains:
                continue

            if not np.all(match_i[match_j]):
                continue

            error_i, error_j = errors[i], errors[j]
            would_subsume = error_i <= error_j * (1 + tolerance)

            diff = pred_diff_on_overlap(pool[i], pool[j])

            records.append(
                {
                    "i": i,
                    "j": j,
                    "overlap_size": int(match_j.sum()),
                    "j_pool_size": int(match_j.sum()),
                    "i_pool_size": int(match_i.sum()),
                    "error_i": error_i,
                    "error_j": error_j,
                    "error_diff": error_i - error_j,
                    "would_subsume": would_subsume,
                    "pred_diff_mean_abs": float(np.mean(np.abs(diff))),
                    "pred_diff_max_abs": float(np.max(np.abs(diff))),
                }
            )

    return pd.DataFrame.from_records(records)


def summarize(df: pd.DataFrame, n_rules_in_pool: int) -> None:
    total_ordered_pairs = n_rules_in_pool * (n_rules_in_pool - 1)
    n_containment_pairs = len(df)
    n_subsumable = int(df["would_subsume"].sum()) if not df.empty else 0
    if not df.empty:
        subsumed_rules = df[df["would_subsume"]]["j"].unique()
        print(f"\nDistinct subsumable rules: {len(subsumed_rules)} / {n_rules_in_pool} " f"({len(subsumed_rules)/n_rules_in_pool:.1%})")

    print("=" * 60)
    print("Subsumption redundancy — preliminary measurement")
    print("=" * 60)
    print(f"Final pool size:                 {n_rules_in_pool}")
    print(f"Total ordered rule pairs:         {total_ordered_pairs}")
    print(f"Containment pairs (mask_i ⊇ mask_j): {n_containment_pairs} " f"({n_containment_pairs / total_ordered_pairs:.2%})")
    print(
        f"Of those, satisfy error_i <= error_j (= actual subsumption): "
        f"{n_subsumable} ({n_subsumable / max(n_containment_pairs, 1):.2%} of containment pairs)"
    )

    if not df.empty:
        subsumable_df = df[df["would_subsume"]]
        if not subsumable_df.empty:
            print("\nPrediction disagreement on overlap, for subsumable pairs:")
            print(subsumable_df["pred_diff_mean_abs"].describe())


def main():
    model = run_single_cycle("airfoil_self_noise", "NA", "spea2")
    pool = get_final_pool(model)

    df = analyze_pool(pool, 0.00)
    n_rules_in_pool = len(pool)
    summarize(df, n_rules_in_pool=n_rules_in_pool)

    out_path = "output/subsumption_pairs_airfoil.csv"
    df.to_csv(out_path, index=False)
    print(f"\nPer-pair records written to {out_path}")


if __name__ == "__main__":
    main()
