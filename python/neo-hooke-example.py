"""Unified Neo-Hookean example for all supported AD backends.

Computes the first and second derivatives of the Neo-Hookean strain energy density
Psi(C) with respect to the right Cauchy-Green deformation tensor C using the
selected automatic differentiation backend. Results are compared against the
analytical reference implementation and runtimes are reported.

Usage
-----
    python neo-hooke-example.py --backend autograd
    python neo-hooke-example.py --backend jax
    python neo-hooke-example.py --backend num-dual
    python neo-hooke-example.py --backend pytorch
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
from neohooke import d2Psi_dC_dC_analytical, dPsi_dC_analytical, psi  # noqa: E402

BACKENDS = ["autograd", "jax", "num-dual", "pytorch"]


def parse_args():
    parser = argparse.ArgumentParser(description="Neo-Hookean AD example")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        required=True,
        help="AD backend to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # import df_dT and d2f_dT2 from the selected backend
    interface_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.backend, "interface.py")
    spec = importlib.util.spec_from_file_location(f"{args.backend}.interface", interface_path)
    interface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(interface)
    df_dT = interface.df_dT
    d2f_dT2 = interface.d2f_dT2

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

    # automatic differentiation
    tic = time.time()
    dPsi_dC_ad = df_dT(lambda C: psi(C, K, G), C)
    d2Psi_dC_dC_ad = d2f_dT2(lambda C: psi(C, K, G), C)
    toc = time.time()
    time_ad = toc - tic

    # summarize results and report errors and time taken
    error_dPsi_dC = np.linalg.norm(dPsi_dC_anltcl - dPsi_dC_ad)
    error_d2Psi_dC_dC = np.linalg.norm(d2Psi_dC_dC_anltcl - d2Psi_dC_dC_ad)
    print(f"Error in dPsi/dC: {error_dPsi_dC:.4e}")
    print(f"Error in d2Psi/dC/dC: {error_d2Psi_dC_dC:.4e}")

    print(f"Time taken for analytical derivatives: {time_analytical:.4e} seconds")
    print(f"Time taken for automatic differentiation with {args.backend}: {time_ad:.4e} seconds")
