# autodiff-tools

Comparison of Python automatic differentiation (AD) libraries for computing first and second derivatives of scalar functions with respect to array inputs, demonstrated on a Neo-Hookean hyperelastic strain energy density.

## Tools

| Tool | Approach | Package |
|---|---|---|
| [autograd](https://github.com/HIPS/autograd) | Operator overloading (reverse-mode) | `autograd` |
| [JAX](https://jax.readthedocs.io) | JIT-compiled reverse-mode AD | `jax` |
| [PyTorch](https://pytorch.org) | Operator overloading (reverse-mode) | `torch` |
| [num-dual](https://github.com/itt-ustutt/num-dual) | (Hyper-)dual numbers (forward-mode) | `num-dual` |

Each tool is wrapped behind a common interface exposing `df_dT(f, T)` and `d2f_dT2(f, T)`. Results are validated against analytical derivatives in `python/utils/neohooke.py`.

## Setup

```bash
pip install -r requirements_pip.txt       # pip
conda install --file requirements_conda.txt  # conda
```

## Usage

Run an example from within its directory:

```bash
cd python/<tool>   # autograd | jax | num-dual | pytorch
python neo-hooke-example.py
```

Each script reports the derivative errors relative to the analytical solution and the runtime of both approaches.
