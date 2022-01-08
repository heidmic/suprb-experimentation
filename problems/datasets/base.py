import pathlib

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from ..base import ProblemRepr

DATASETS_PATH = (pathlib.Path(__file__).parent / 'data').resolve()


def load_dataset(filename: str, target_column: str, return_X_y: bool, as_frame: bool) -> ProblemRepr:
    frame = pd.read_csv(DATASETS_PATH / filename, sep=',')

    data = frame.loc[:, frame.columns != target_column]
    target = frame[target_column]

    if not as_frame:
        data = data.to_numpy(dtype=np.float)
        target = target.to_numpy(dtype=np.float)

    if return_X_y:
        return data, target
    else:
        if as_frame:
            return Bunch(frame=frame, data=data, target=target)
        else:
            return Bunch(X=data, y=target)


def load_combined_cycle_power_plant(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Combined Cycle Power Plant dataset.

    ==============   ==================
    Samples total    9568
    Dimensionality   4
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant.
    """
    return load_dataset(filename='ccpp.csv', target_column='PE', return_X_y=return_X_y, as_frame=as_frame)


def load_gas_turbine(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Gas Turbine dataset.

    ==============   ==================
    Samples total    36733
    Dimensionality   10
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set.
    """
    return load_dataset(filename='gas_turbine.csv', target_column='TEY', return_X_y=return_X_y, as_frame=as_frame)


def load_concrete_strength(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Concrete Strength dataset.

    ==============   ==================
    Samples total    1030
    Dimensionality   8
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength.
    """
    return load_dataset(filename='concrete.csv', target_column='CCS', return_X_y=return_X_y, as_frame=as_frame)


def load_airfoil_self_noise(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Airfoil Self-Noise dataset.

    ==============   ==================
    Samples total    1503
    Dimensionality   5
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise.
    """
    return load_dataset(filename='airfoil_self_noise.csv', target_column='SPL', return_X_y=return_X_y,
                        as_frame=as_frame)
