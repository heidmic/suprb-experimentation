import pathlib

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from ..base import ProblemRepr

from sklearn.datasets import load_diabetes

DATASETS_PATH = (pathlib.Path(__file__).parent / 'data').resolve()
CLASS_DATASETS_PATH = (pathlib.Path(__file__).parent / 'class_data').resolve()


def load_dataset(filename: str, target_column: str, return_X_y: bool, as_frame: bool,
                 remove_columns: list = None) -> ProblemRepr:
    frame = pd.read_csv(DATASETS_PATH / filename, sep=',')

    data = frame.drop(columns=[target_column] + (remove_columns if remove_columns is not None else []))
    target = frame[target_column]

    if not as_frame:
        data = data.to_numpy(dtype=float)
        target = target.to_numpy(dtype=float)

    if return_X_y:
        return data, target
    elif as_frame:
        return Bunch(frame=frame, data=data, target=target)
    else:
        return Bunch(X=data, y=target)
    

def load_class_dataset(filename: str, target_column: str, return_X_y: bool, as_frame: bool,
                 remove_columns: list = None, label_to_num: bool = True) -> ProblemRepr:
    frame = pd.read_csv(CLASS_DATASETS_PATH / filename, sep=',')

    data = frame.drop(columns=[target_column] + (remove_columns if remove_columns is not None else []))
    target = frame[target_column]

    if not as_frame:
        data = data.to_numpy(dtype=float)
        target = target.to_numpy()
    
    if label_to_num:
        labels = np.unique(y)
        toNum = dict(zip(labels, range(len(labels))))
        y = [toNum[x[0]] for x in y]
    
    if return_X_y:
        return data, target
    elif as_frame:
        return Bunch(frame=frame, data=data, target=target)
    else:
        return Bunch(X=data, y=target)


def load_iris(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Combined Cycle Power Plant dataset.

    ==============   ==================
    Samples total    150
    Dimensionality   4
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/53/iris.
    """
    return load_class_dataset(filename='iris.csv', target_column='y', return_X_y=return_X_y, as_frame=as_frame)

def load_breastcancer(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Combined Cycle Power Plant dataset.

    ==============   ==================
    Samples total    569
    Dimensionality   30
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.
    """
    return load_class_dataset(filename='breastcancer.csv', target_column='Y', return_X_y=return_X_y, as_frame=as_frame)


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


def load_energy_heat(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Energy efficiency dataset with heating load as target.

    ==============   ==================
    Samples total    768
    Dimensionality   8
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/energy+efficiency.
    """

    return load_dataset(filename='energy.csv', target_column='Y1', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['Y2'])


def load_energy_cool(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Energy efficiency dataset with cooling load as target.

    ==============   ==================
    Samples total    768
    Dimensionality   8
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/energy+efficiency.
    """

    return load_dataset(filename='energy.csv', target_column='Y2', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['Y1'])


def load_forest_fires(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Forest Fires dataset.

    ==============   ==================
    Samples total    517
    Dimensionality   13
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/forest+fires.
    """

    return load_dataset(filename='forest_fires.csv', target_column='area', return_X_y=return_X_y,
                        as_frame=as_frame)


def load_parkinson_total(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Parkinson dataset with total UPDRS as target.

    ==============   ==================
    Samples total    5875
    Dimensionality   26
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring.
    """

    return load_dataset(filename='parkinson.csv', target_column='total_UPDRS', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['subject#', 'test_time', 'motor_UPDRS'])


def load_parkinson_motor(return_X_y: bool = True, as_frame: bool = False):
    """ Load and return the Parkinson dataset with motor UPDRS as target.

    ==============   ==================
    Samples total    5875
    Dimensionality   26
    Features         real, TODO: ranges
    Targets          real, TODO: ranges
    ==============   ==================

    Downloaded from https://archive.ics.uci.edu/ml/datasets/parkinsons+telemonitoring.
    """

    return load_dataset(filename='parkinson.csv', target_column='motor_UPDRS', return_X_y=return_X_y,
                        as_frame=as_frame, remove_columns=['subject#', 'test_time', 'total_UPDRS'])


def load_protein_structure(return_X_y: bool = True, as_frame: bool = False, sample: bool = True):
    """ Load and return the Protein Structure dataset.
        ==============   ==================
        Samples total    45730
        Dimensionality   9
        Features         real, TODO: ranges
        Targets          real, TODO: ranges
        ==============   ==================
        Downloaded from
        https://https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure.
        """
    return load_dataset(filename='protein_structure.csv', target_column='RMSD', return_X_y=return_X_y,
                        as_frame=as_frame)
