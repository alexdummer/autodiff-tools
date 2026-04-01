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

Run the unified example from the `python/` directory, selecting the desired backend:

```bash
cd python
python neo-hooke-example.py --backend autograd   # operator-overloading reverse-mode AD
python neo-hooke-example.py --backend jax        # JIT-compiled reverse-mode AD
python neo-hooke-example.py --backend num-dual   # forward-mode AD via dual numbers
python neo-hooke-example.py --backend pytorch    # operator-overloading reverse-mode AD
```

Each run reports the derivative errors relative to the analytical solution and the runtime of both approaches.
