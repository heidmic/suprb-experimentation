from typing import Union

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import Bunch

ProblemRepr = Union[tuple[np.ndarray, np.ndarray], Bunch]


def scale_X_y(X: np.ndarray, y: np.ndarray, feature_range: tuple = (-1, 1), return_X_y=True) -> ProblemRepr:
    """Scale and transform X with MinMaxScaler and y with StandardScaler."""
    X_scaler = MinMaxScaler(feature_range=feature_range)
    y_scaler = StandardScaler()

    X = X_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape((-1, 1))).reshape((-1,))

    if return_X_y:
        return X, y, y_scaler
    else:
        return Bunch(X=X, y=y, X_scaler=X_scaler, y_scaler=y_scaler)
