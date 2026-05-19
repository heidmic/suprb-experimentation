import numpy as np

from .base import with_bounds


@with_bounds([[-0.5, 2.5]])
def gramacy_lee_N1(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/grlee12.html"""
    (x,) = X.T
    return (np.sin(10 * np.pi * x) / (2 * x)) + (x - 1) ** 4


@with_bounds([[-2, 6]] * 2)
def gramacy_lee_N2(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/grlee08.html"""
    x1, x2 = X.T
    return x1 * np.exp(-(x1**2) - x2**2)


@with_bounds([[-2, 6]] * 6)
def gramacy_lee_N3(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/grlee09.html"""
    x1, x2, x3, x4, _, _ = X.T
    return np.exp(np.sin((0.9 * (x1 + 0.48)) ** 10)) + x2 * x3 + x4


@with_bounds([[0, 20]])
def higdon_gramacy_lee(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/hig02grlee08.html"""
    (x,) = X.T
    return np.where(x < 10, np.sin(np.pi * x / 5) + 0.2 * np.cos(4 * np.pi * x / 5), x / 10 - 1)


@with_bounds([[-1, 1]] * 2)
def frog(X: np.ndarray) -> np.ndarray:
    """Taken from the original SupRB-2 repository."""
    x = X[:, 0] + 1
    a = X[:, 1]

    indices = x + a <= 1
    inter = x + a

    y = np.empty(X.shape[0])

    y[indices] = inter[indices]
    y[~indices] = (2 - inter)[~indices]

    return y


@with_bounds([[0, 1]] * 2)
def lim(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/limetal02non.html"""
    x1, x2 = X.T
    return 1 / 6 * ((30 + 5 * x1 * np.sin(5 * x1)) * (4 + np.exp(-5 * x2)) - 100)


@with_bounds([[0, 1]] * 2)
def franke(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/franke2d.html"""
    x1, x2 = X.T
    y = (
        0.75 * np.exp(-((9 * x1 - 2) ** 2) / 4 - ((9 * x2 - 2) ** 2) / 4)
        + 0.75 * np.exp(-((9 * x1 + 1) ** 2) / 49 - (9 * x2 + 1) / 10)
        + 0.5 * np.exp(-((9 * x1 - 7) ** 2) / 4 - ((9 * x2 - 3) ** 2) / 4)
        - 0.2 * np.exp(-((9 * x1 - 4) ** 2) - (9 * x2 - 7) ** 2)
    )
    return y


@with_bounds([[-0.5, 0.5]] * 20)
def welch(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/welchetal92.html"""
    return (
        (5 * X[:, 11] / 1 + X[:, 0])
        + 5 * (X[:, 3] - X[:, 19]) ** 2
        + X[:, 4]
        + 40 * X[:, 18] ** 3
        - 5 * X[:, 18]
        + 0.05 * X[:, 1]
        + 0.08 * X[:, 2]
        - 0.03 * X[:, 2]
        + 0.03 * X[:, 6]
        - 0.09 * X[:, 8]
        - 0.01 * X[:, 9]
        - 0.07 * X[:, 10]
        + 0.25 * X[:, 12] ** 2
        - 0.04 * X[:, 13]
        + 0.06 * X[:, 14]
        - 0.01 * X[:, 16]
        - 0.03 * X[:, 17]
    )


@with_bounds([[0, 1]] * 8)
def dette_pepelyshev(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/detpep108d.html"""
    s = np.sum([i * np.log(1 + np.sum(X[:, 0:i])) for i in range(3, 8)])
    y = (
        4 * (X[:, 0] - 2 + 8 * X[:, 1] - 8 * X[:, 1] ** 2) ** 2
        + (3 - 4 * X[:, 1])
        + 16 * np.sqrt(X[:, 2] + 1) * (2 * X[:, 2] - 1) ** 2
        + s
    )
    return y


@with_bounds([[0, 1]] * 5)
def friedman(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/fried.html"""
    x1, x2, x3, x4, x5 = X.T
    return 10 * np.sin(np.pi * x1 * x2) + 20 * (x3 - 0.5) ** 2 + 10 * x4 + 5 * x5


@with_bounds(
    [
        [30, 60],  # (1) M
        [0.005, 0.020],  # (2) S
        [0.002, 0.010],  # (3) V_0
        [1_000, 5_000],  # (4) k
        [90_000, 110_000],  # (5) P_0
        [290, 296],  # (6) T_a
        [340, 360],  # (7) T_0
    ]
)
def piston_simulation(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/piston.html"""
    M, S, V_0, k, P_0, T_a, T_0 = X.T
    A = P_0 * S + 19.62 * M - (k * V_0) / S
    prod = (P_0 * V_0) / T_0
    V = S / (2 * k) * (np.sqrt(A**2 + 4 * k * prod * T_a) - A)
    y = 2 * np.pi * np.sqrt(M / (k + S**2 * prod * T_a / (V**2)))
    return y


@with_bounds(
    [
        [50, 150],  # (1) R_b1
        [25, 70],  # (2) R_b2
        [0.5, 3],  # (3) R_f
        [1.2, 2.5],  # (4) R_c1
        [0.25, 1.2],  # (5) R_c2
        [50, 300],  # (6) beta
    ]
)
def otl_circuit(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/otlcircuit.html"""
    R_b1, R_b2, R_f, R_c1, R_c2, beta = X.T
    V_b1 = 12 * R_b2 / (R_b1 + R_b2)
    c = beta * (R_c2 + 9)
    d = c + R_f
    return ((V_b1 + 0.74) * c) / d + (11.25 * R_f) / d + (0.74 * R_f * c) / (d * R_c1)


@with_bounds(
    [
        [0.05, 0.15],  # (1) r_w
        [100, 50_000],  # (2) r
        [63_070, 115_600],  # (3) T_u
        [990, 1110],  # (4) H_u
        [63.1, 116],  # (5) T_l
        [700, 820],  # (6) H_l
        [1120, 1680],  # (7) L
        [9855, 12_045],  # (8) K_w
    ]
)
def borehole(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/borehole.html"""
    r_w, r, T_u, H_u, T_l, H_l, L, K_w = X.T
    y = (2 * np.pi * T_u * (H_u - H_l)) / (
        np.log(r / r_w) * (1 + ((2 * L * T_u) / (np.log(r / r_w) * r_w**2 * K_w)) + T_u / T_l)
    )
    return y


def deg2rad(deg):
    rad = deg / 180.0 * np.pi
    return rad


@with_bounds(
    [
        [150, 200],  # (1) S_w
        [220, 300],  # (2) W_fw
        [6, 10],  # (3) A
        [-10, 10],  # (4) lambda_maj
        [16, 45],  # (5) q
        [0.5, 1],  # (6) lambda_min
        [0.08, 0.18],  # (7) t_c
        [2.5, 6],  # (8) N_z
        [1700, 2500],  # (9) W_dg
        [0.025, 0.08],  # (10) W_p
    ]
)
def wing_weight(X: np.ndarray) -> np.ndarray:
    """https://www.sfu.ca/~ssurjano/wingweight.html"""
    S_w, W_fw, A, lambda_maj, q, lambda_min, t_c, N_z, W_dg, W_p = X.T
    lambda_maj = deg2rad(lambda_maj)
    y = (
        0.036 * S_w**0.758 * W_fw**0.0035 * (A / (np.cos(lambda_maj) ** 2)) ** 0.6 * q**0.006 * lambda_min**0.04
        + ((100 * t_c) / np.cos(lambda_maj)) ** -0.3 * (N_z * W_dg) ** 0.49
        + S_w * W_p
    )
    return y
