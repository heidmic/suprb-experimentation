import numpy as np

from .base import with_bounds


@with_bounds([-2.5, 7])
def poly3(X: np.ndarray) -> np.ndarray:
    """Taken from the original SupRB-2 repository."""
    return np.sum(0.75 * X**3 - 5 * X**2 + 4 * X + 12, axis=1)


@with_bounds([-1, 1])
def linear(X: np.ndarray) -> np.ndarray:
    return np.sum(X, axis=1)


@with_bounds([0, 2 * np.pi])
def sin(X: np.ndarray) -> np.ndarray:
    return np.sum(np.sin(X), axis=1)


@with_bounds([-10, 10])
def sphere(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/spheref.html"""
    return np.sum(X**2, axis=1)


@with_bounds([0, 10])
def alpine_N2(X: np.ndarray) -> np.ndarray:
    """http://benchmarkfcns.xyz/benchmarkfcns/alpinen2fcn.html"""
    return np.prod(np.sqrt(X) * np.sin(X), axis=1)


@with_bounds([-20, 20])
def ackley_N4(X: np.ndarray) -> np.ndarray:
    """http://benchmarkfcns.xyz/benchmarkfcns/ackleyn4fcn.html"""
    shifted = np.roll(X, shift=1, axis=1)
    return np.sum(np.e**0.2 + np.sqrt(X**2 + shifted**2) + 3 * np.cos(2 * X) + np.sin(2 * shifted), axis=1)


@with_bounds([-5, 5])
def styblinski_tang(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/stybtang.html"""
    return 0.5 * np.sum((X**4) - 16 * (X**2) + 5 * X, axis=1)


@with_bounds([-1, 1])
def exponential(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/boha.html"""
    return -np.exp(-0.5 * np.sum(X**2, axis=1))
