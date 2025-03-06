from __future__ import annotations

from abc import abstractmethod, ABCMeta
from numbers import Integral
from typing import Iterable

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import get_scorer, mean_squared_error, accuracy_score
from sklearn.model_selection import cross_validate, KFold

from suprb.logging.default import DefaultLogger

def check_scoring(scoring):
    """Always use R^2 and MSE for evaluation."""

    if scoring is None:
        scoring = set()
    elif isinstance(scoring, Iterable):
        scoring = set(scoring)
    else:
        scoring = {scoring}
    scoring.update({'r2', 'neg_mean_squared_error'})

    return list(scoring)

def check_class_scoring(scoring):
    """Always use R^2 and MSE for evaluation."""

    if scoring is None:
        scoring = set()
    elif isinstance(scoring, Iterable):
        scoring = set(scoring)
    else:
        scoring = {scoring}
    scoring.update({'f1', 'accuracy'})

    return list(scoring)

def check_cv(cv, random_state=None):
    """Enable shuffle by default."""
    if isinstance(cv, Integral):
        cv = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    return cv


class Evaluation(metaclass=ABCMeta):

    def __init__(self, estimator: BaseEstimator, random_state: int, verbose: int):
        self.estimator = estimator
        self.random_state = random_state
        self.verbose = verbose

    @abstractmethod
    def __call__(self, params: dict, **kwargs) -> tuple[list[BaseEstimator], dict]:
        pass

class CustomUnfitEvaluation(Evaluation, metaclass=ABCMeta):
    
    def __init__(
            self,
            dummy_estimator: BaseEstimator,
            X: np.ndarray,
            y: np.ndarray,
            random_state: int = None,
            verbose: int = 0,
            local_model: BaseEstimator = None,
            trained_estimators: list[BaseEstimator] = None,
            isClass: bool = False
    ):
        super().__init__(estimator=dummy_estimator, random_state=random_state, verbose=verbose)
        self.X = X
        self.y = y
        self.local_model = local_model
        self.trained_estimators = trained_estimators
        self.isClass = isClass
    
    def __call__(self, **kwargs) -> tuple[list[BaseEstimator], dict]:
        cv = check_cv(kwargs.pop('cv', None), random_state=self.random_state)
        scores = []
        estimators = []
        for i, (train_index, test_index) in enumerate(cv.split(self.X)):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            estimator = self.trained_estimators[i]
            estimator.model_swap_fit(self.local_model,X_train, y_train)
            estimator.logger_ = DefaultLogger()
            estimator.logger_.log_init(X_train, y_train, estimator)
            estimator.logger_.log_final(X_train, y_train, estimator)
            prediction = estimator.predict(X_test)
            scorer = mean_squared_error if not self.isClass else accuracy_score
            scores.append(scorer(y_test, prediction))
            estimators.append(estimator)
        return estimators, {'test_score': [scores]}

class BaseCrossValidate(Evaluation, metaclass=ABCMeta):
    estimators_: list[BaseEstimator]
    results_: dict

    def cross_validate(self, X: np.ndarray, y: np.ndarray, params: dict, **kwargs):
        #scoring = check_scoring(kwargs.pop('scoring', None))
        scoring = kwargs.pop('scoring', None)
        cv = check_cv(kwargs.pop('cv', None), random_state=self.random_state)

        estimator = clone(self.estimator)
        estimator.set_params(**params)

        # Do cross-validation
        scores = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring=scoring,
            cv=cv,
            return_estimator=True,
            verbose=self.verbose,
            **kwargs
        )

        return scores


class CrossValidateTest(BaseCrossValidate):
    """Evaluate the estimator using cross validation and an extra test set."""

    def __init__(
            self,
            estimator: BaseEstimator,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_test: np.ndarray,
            y_test: np.ndarray,
            random_state: int = None,
            verbose: int = 0,
    ):
        super().__init__(estimator=estimator, random_state=random_state, verbose=verbose)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __call__(self, **kwargs) -> tuple[list[BaseEstimator], dict]:

        scores = self.cross_validate(self.X_train, self.y_train, **kwargs)

        # Save estimators externally
        estimators = scores.pop('estimator')

        # Rename test_scores to val_scores, because we have an additional test set
        new_scores = {}
        for key, value in scores.items():
            if key.startswith('test_'):
                scoring = key.removeprefix('test_')
                new_scores['val_' + scoring] = scores[key]
                scorer = get_scorer(scoring)
                new_scores['test_' + scoring] = np.array(
                    [scorer(estimator, self.X_test, self.y_test) for estimator in estimators])
            else:
                new_scores[key] = value

        self.estimators_, self.results_ = estimators, new_scores
        return estimators, new_scores


class CrossValidate(BaseCrossValidate):
    """Evaluate the estimator using cross validation."""

    def __init__(
            self,
            estimator: BaseEstimator,
            X: np.ndarray,
            y: np.ndarray,
            random_state: int = None,
            verbose: int = 0,
    ):
        super().__init__(estimator=estimator, random_state=random_state, verbose=verbose)
        self.X = X
        self.y = y

    def __call__(self, **kwargs) -> tuple[list[BaseEstimator], dict]:
        scores = self.cross_validate(self.X, self.y, **kwargs)

        # Save estimators externally
        estimators = scores.pop('estimator')

        self.estimators_, self.results_ = estimators, scores
        return estimators, scores
