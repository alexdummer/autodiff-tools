from collections.abc import Callable

import numpy as np
import torch


def df_dT(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the first derivative of a function f with respect to T using PyTorch automatic differentiation.
    Args:
        f: function that takes a PyTorch tensor as input and returns a scalar tensor output
        T: numpy array of input values
    Returns:
        df_dT: numpy array of the same shape as T containing the first derivatives of f with respect to T
    """
    T_torch = torch.tensor(T, dtype=torch.float64)
    result = torch.autograd.functional.jacobian(f, T_torch)
    return result.numpy()


def d2f_dT2(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the second derivative of a function f with respect to T using PyTorch automatic differentiation.
    Args:
        f: function that takes a PyTorch tensor as input and returns a scalar tensor output
        T: numpy array of input values
    Returns:
        d2f_dT2: numpy array of shape (*T.shape, *T.shape) containing the second derivatives of f with respect to T
    """
    T_torch = torch.tensor(T, dtype=torch.float64)
    result = torch.autograd.functional.hessian(f, T_torch)
    return result.numpy()
