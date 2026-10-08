from typing import Union, Callable

import numpy as np
from sklearn.utils import Bunch, shuffle as apply_shuffle
from suprb.utils import check_random_state

from ..base import scale_X_y, ProblemRepr


def load_test_function(
    func: Callable,
    bounds: np.ndarray = None,
    n_samples: int = 1000,
    n_dims: int = None,
    scale=True,
    noise: float = 0,
    shuffle=True,
    random_state: int = None,
    return_X_y=True,
) -> ProblemRepr:
    """Generate X and y from a custom test function.

    Parameters
        ----------
        func : Callable
            Function or object that defines a `bounds` attribute and can be called with `__call__`.
        bounds: np.ndarray
        n_samples: int
        n_dims: int
        scale: {bool, tuple, list, np.ndarray}
        noise: float
        shuffle: bool
        random_state:  int
        return_X_y: bool
    """

    random_state_ = check_random_state(random_state)

    # Sample the input space
    if bounds is not None:
        low, high = bounds.T
    else:
        low, high = func.bounds.T

    # Expand the bounds to input dims, if wanted and possible
    if n_dims is not None and func.bounds.ndim == 1:
        low, high = np.hstack([high] * n_dims), np.hstack([low] * n_dims)
    else:
        n_dims = func.bounds.T.shape[1]

    X = random_state_.uniform(low, high, size=(n_samples, n_dims))
    y = func(X)

    # Scale the input and output space
    if scale:
        if not isinstance(scale, bool):
            scale_result = scale_X_y(X, y, feature_range=scale, return_X_y=return_X_y)
        else:
            scale_result = scale_X_y(X, y, return_X_y=return_X_y)

        if return_X_y:
            X, y = scale_result
        else:
            X, y = scale_result.X, scale_result.y

    # Add noise
    if noise > 0:
        y += random_state_.normal(scale=noise, size=n_samples)

    # Shuffle both sets
    if shuffle:
        # Note that we have to supply `random_state` (the parameter) to `apply_shuffle` here,
        # and not `random_state_` (the `np.random.Generator` instance)
        # because `sklearn.utils.check_random_state()` does currently not support `np.random.Generator`.
        # See https://github.com/scikit-learn/scikit-learn/issues/16988 for the current status.
        # Our `suprb.utils.check_random_state()` can handle `np.random.Generator`.
        X, y = apply_shuffle(X, y, random_state=random_state)

    # Return arrays directly or use Bunch, depending on parameter
    if return_X_y:
        return X, y
    elif scale:
        return Bunch(X=X, y=y, X_scaler=scale_result.X_scaler, y_scaler=scale_result.y_scaler)
    else:
        return Bunch(X=X, y=y)


def with_bounds(bounds: list):
    def wrapper(func):
        func.bounds = np.array(bounds, dtype=float)
        return func

    return wrapper
