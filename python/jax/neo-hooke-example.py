import sys
import time

import numpy as np
from interface import d2f_dT2_jax, df_dT_jax

sys.path.append("../utils")
from neohooke import d2Psi_dC_dC_analytical, dPsi_dC_analytical, psi  # noqa: E402

if __name__ == "__main__":

    # elastic constants
    K = 3500
    G = 1500

    # deformation gradient and right Cauchy-Green deformation tensor
    F = np.identity(3)
    F[0, 0] = 1.2
    F[0, 1] = 0.2
    C = F.T @ F

    # analytical computations
    tic = time.time()
    dPsi_dC_anltcl = dPsi_dC_analytical(C, K, G)
    d2Psi_dC_dC_anltcl = d2Psi_dC_dC_analytical(C, K, G)
    toc = time.time()
    time_analytical = toc - tic

    # JAX automatic differentiation
    tic = time.time()
    dPsi_dC_jax = df_dT_jax(lambda C: psi(C, K, G), C)
    d2Psi_dC_dC_jax = d2f_dT2_jax(lambda C: psi(C, K, G), C)
    toc = time.time()
    time_jax = toc - tic

    # summarize results only report errors and time taken for numerical derivatives
    error_dPsi_dC = np.linalg.norm(dPsi_dC_anltcl - dPsi_dC_jax)
    error_d2Psi_dC_dC = np.linalg.norm(d2Psi_dC_dC_anltcl - d2Psi_dC_dC_jax)
    print(f"Error in dPsi/dC: {error_dPsi_dC:.4e}")
    print(f"Error in d2Psi/dC/dC: {error_d2Psi_dC_dC:.4e}")

    # format the time with scientific format taken for both analytical and numerical derivatives
    print(f"Time taken for analytical derivatives: {time_analytical:.4e} seconds")
    print(f"Time taken for automatic differentiation with JAX: {time_jax:.4e} seconds")
