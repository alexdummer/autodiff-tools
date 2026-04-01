from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def df_dT_jax(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the first derivative of a function f with respect to T using JAX automatic differentiation.
    Args:
        f: function that takes a JAX array as input and returns a scalar output
        T: numpy array of input values
    Returns:
        df_dT: numpy array of the same shape as T containing the first derivatives of f with respect to T
    """
    T_jax = jnp.array(T)
    return np.array(jax.grad(f)(T_jax))


def d2f_dT2_jax(f: Callable, T: np.ndarray) -> np.ndarray:
    """Computes the second derivative of a function f with respect to T using JAX automatic differentiation.
    Args:
        f: function that takes a JAX array as input and returns a scalar output
        T: numpy array of input values
    Returns:
        d2f_dT2: numpy array of shape (*T.shape, *T.shape) containing the second derivatives of f with respect to T
    """
    T_jax = jnp.array(T)
    return np.array(jax.hessian(f)(T_jax))
