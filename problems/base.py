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

    mean = y_scaler.mean_
    var = y_scaler.var_
    std = np.sqrt(y_scaler.var_)

    mae = [-0.13564742,
           -0.34432709,
           -0.1389487,
           -0.13933356,
           -0.36175137,
           -0.36024185,
           -0.13422692,
           -0.13851665,
           -0.35299664,
           -0.3673267,
           -0.14082466,
           -0.35727883,
           -0.14139092,
           -0.13866312,
           -0.13748454,
           -0.35749145,
           -0.36459038,
           -0.35303291,
           -0.34432222,
           -0.13787621,
           -0.13644293,
           -0.36040402,
           -0.34968627,
           -0.13812453,
           -0.13839405,
           -0.34862736,
           -0.34712178,
           -0.13810488,
           -0.36025236,
           -0.35279057,
           -0.13910158,
           -0.36323381,
           -0.36422407,
           -0.13744483,
           -0.13399331,
           -0.35976951,
           -0.1359362,
           -0.13492696,
           -0.13655849,
           -0.13490975,
           -0.35618476,
           -0.3582211]

    mse = [-0.03594984,
           -0.26604244,
           -0.03600888,
           -0.03591213,
           -0.28806203,
           -0.30248924,
           -0.03419983,
           -0.03516302,
           -0.26856837,
           -0.27935014,
           -0.03711153,
           -0.28123374,
           -0.03878871,
           -0.03682399,
           -0.03642196,
           -0.27459767,
           -0.28696817,
           -0.27168539,
           -0.26160585,
           -0.03733634,
           -0.03553954,
           -0.28353648,
           -0.26538112,
           -0.03575478,
           -0.03543022,
           -0.27298205,
           -0.26308385,
           -0.03700736,
           -0.27296654,
           -0.26081759,
           -0.0370265,
           -0.32783466,
           -0.3078942,
           -0.03551542,
           -0.03387359,
           -0.30890035,
           -0.0354787,
           -0.03525432,
           -0.0360248,
           -0.03467181,
           -0.26980094,
           -0.43951638]

    print(mean, var, std)

    mse = np.array(mse)
    mae = np.array(mae)

    cleared_mse = mse*var
    cleared_mae = mae*std

    print("MSE:", cleared_mse)
    print("MAE:", cleared_mae)
    exit()

    if return_X_y:
        return X, y
    else:
        return Bunch(X=X, y=y, X_scaler=X_scaler, y_scaler=y_scaler)
