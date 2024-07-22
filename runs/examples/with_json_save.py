import numpy as np
import click

from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit

from experiments.evaluation import CrossValidate

from problems import scale_X_y
from experiments import Experiment


from suprb import SupRB
from suprb.optimizer.rule.es import ES1xLambda
from suprb.optimizer.solution.ga import GeneticAlgorithm
import suprb.json as suprb_json


random_state = 42


def load_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    method_name = f"load_{name}"
    from problems import datasets
    if hasattr(datasets, method_name):
        return getattr(datasets, method_name)(**kwargs)


@click.command()
@click.option('-p', '--problem', type=click.STRING, default='airfoil_self_noise')
@click.option('-j', '--job_id', type=click.STRING, default='NA')
def run(problem: str, job_id: str):
    print(f"Problem is {problem}, with job id {job_id}")

    X, y = load_dataset(name=problem, return_X_y=True)
    X, y = scale_X_y(X, y)
    X, y = shuffle(X, y, random_state=random_state)

    estimator = SupRB(rule_generation=ES1xLambda(n_iter=2,
                                             lmbda=2,
                                             operator='+',
                                             delay=150,
                                             random_state=random_state,
                                             n_jobs=1),
                  solution_composition=GeneticAlgorithm(n_iter=2,
                                                        population_size=2,
                                                        elitist_ratio=0.2,
                                                        random_state=random_state,
                                                        n_jobs=1))


    experiment_name = f'{problem}'
    print(experiment_name)
    experiment = Experiment(name=experiment_name,  verbose=10)


    random_states = np.random.SeedSequence(random_state).generate_state(8)
    experiment.with_random_states(random_states, n_jobs=8)

    evaluation = CrossValidate(
        estimator=estimator, X=X, y=y, random_state=random_state, verbose=10)

    experiment.perform(evaluation, cv=ShuffleSplit(
        n_splits=8, test_size=0.25, random_state=random_state), n_jobs=8)
 

if __name__ == '__main__':
    run()
