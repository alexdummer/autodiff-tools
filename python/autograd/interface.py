from collections.abc import Callable

import autograd
import numpy as np


def df_dT(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the first derivative of a function f with respect to T using autograd automatic differentiation.
    Args:
        f: function that takes a numpy array as input and returns a scalar output
        T: numpy array of input values
    Returns:
        df_dT: numpy array of the same shape as T containing the first derivatives of f with respect to T
    """
    return autograd.grad(f)(T.astype(np.float64))


def d2f_dT2(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the second derivative of a function f with respect to T using autograd automatic differentiation.
    Args:
        f: function that takes a numpy array as input and returns a scalar output
        T: numpy array of input values
    Returns:
        d2f_dT2: numpy array of shape (*T.shape, *T.shape) containing the second derivatives of f with respect to T
    """
    return autograd.jacobian(autograd.grad(f))(T.astype(np.float64))
