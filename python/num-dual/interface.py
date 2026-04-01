from collections.abc import Callable

import numpy as np
from num_dual import Dual64 as D64
from num_dual import HyperDual64 as HD64


def makeDual(T: np.ndarray) -> np.ndarray:
    """Converts a numpy array T into an array of dual numbers with the same shape.
    Args:
        T: numpy array of input values
    Returns:
        T_d: numpy array of dual numbers with the same shape as T
    """
    # prepare an array of dual numbers with the same shape as T
    T_d = np.empty(T.shape, dtype=D64)

    for i in range(T.size):
        T_d.flat[i] = D64(T.flat[i], 0.0)

    return T_d


def makeHyperDual(T: np.ndarray) -> np.ndarray:
    """Converts a numpy array T into an array of hyper dual numbers with the same shape.
    Args:
        T: numpy array of input values
    Returns:
        T_hd: numpy array of hyper dual numbers with the same shape as T
    """
    # prepare an array of hyper dual numbers with the same shape as T
    T_hd = np.empty(T.shape, dtype=HD64)

    # initialize the hyper dual numbers with the values from T and zero for the derivatives
    for i in range(T.size):
        T_hd.flat[i] = HD64(T.flat[i], 0.0, 0.0, 0.0)
    return T_hd


def df_dT_num_dual(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the first derivative of a function f with respect to T using dual numbers.
    Args:
        f: function that takes a numpy array as input and returns a scalar output
        T: numpy array of input values
    Returns:
        df_dT: numpy array of the same shape as T containing the first derivatives of f with respect to T
    """
    # make a dual number array from T to compute first derivatives
    T_d = makeDual(T)
    # initialize an array to store the first derivatives
    df_dT = np.empty(T_d.shape, dtype=np.float64)
    # perturbation value for the first derivative
    eps = D64(0, 1.0)

    for i in range(T_d.size):
        # pertubate the i-th component of T_d
        T_d.flat[i] += eps
        # compute the function value
        f_d = f(T_d)
        # extract the first derivative and store it in the corresponding position in df_dT
        df_dT.flat[i] = f_d.first_derivative
        # reset the i-th component of T_d
        T_d.flat[i] -= eps

    return df_dT


def d2f_dT2_num_dual(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the second derivative of a function f with respect to T using hyper dual numbers.
    Args:
        f: function that takes a numpy array as input and returns a scalar output
        T: numpy array of input values
    Returns:
        d2f_dT2: numpy array of shape (T.size, T.size) containing the second derivatives of f with respect to T
    """
    # make a hyper dual number array from T to compute second derivatives
    T_hd = makeHyperDual(T)

    # initialize an array to store the second derivatives
    d2f_dT2 = np.empty((*T_hd.shape, *T_hd.shape), dtype=np.float64)

    # perturbation values for the first and second derivatives
    eps1 = HD64(0, 1.0, 0.0, 0.0)
    eps2 = HD64(0, 0.0, 1.0, 0.0)

    for i in range(T_hd.size):
        # pertubate the i-th component of T_hd for the first derivative
        T_hd.flat[i] += eps1
        for j in range(T_hd.size):
            # pertubate the j-th component of T_hd for the second derivative
            T_hd.flat[j] += eps2
            # compute the function value
            f_hd = f(T_hd)
            # index in the flattened array
            idx = i * T_hd.size + j
            # extract the second derivative
            d2f_dT2.flat[idx] = f_hd.second_derivative
            # reset the j-th component of T_hd
            T_hd.flat[j] -= eps2

        # reset the i-th component of T_hd
        T_hd.flat[i] -= eps1

    return d2f_dT2
