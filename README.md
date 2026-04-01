# autodiff-tools

A comparison of Python automatic differentiation (AD) libraries for computing first and second derivatives of scalar functions with respect to array inputs.

## Overview

This repository demonstrates how to use four different AD tools to compute derivatives of a **Neo-Hookean hyperelastic strain energy density** function with respect to the right Cauchy-Green deformation tensor **C**:

| Tool | Approach | Package |
|---|---|---|
| [autograd](https://github.com/HIPS/autograd) | Operator overloading (reverse-mode) | `autograd` |
| [JAX](https://jax.readthedocs.io) | JIT-compiled reverse-mode AD | `jax` |
| [PyTorch](https://pytorch.org) | Operator overloading (reverse-mode) | `torch` |
| [num-dual](https://github.com/itt-ustutt/num-dual) | (Hyper-)dual numbers (forward-mode) | `num-dual` |

Each tool is wrapped behind a common interface that provides:

- `df_dT(f, T)` – first derivative of `f` with respect to `T`
- `d2f_dT2(f, T)` – second derivative of `f` with respect to `T`

Results are validated against analytical derivatives implemented in `python/utils/neohooke.py`.

## Repository Structure

```
autodiff-tools/
├── python/
│   ├── autograd/
│   │   ├── interface.py          # autograd wrapper
│   │   └── neo-hooke-example.py  # runnable example
│   ├── jax/
│   │   ├── interface.py          # JAX wrapper
│   │   └── neo-hooke-example.py  # runnable example
│   ├── num-dual/
│   │   ├── interface.py          # num-dual wrapper
│   │   └── neo-hooke-example.py  # runnable example
│   ├── pytorch/
│   │   ├── interface.py          # PyTorch wrapper
│   │   └── neo-hooke-example.py  # runnable example
│   └── utils/
│       └── neohooke.py           # analytical reference implementation
├── requirements_pip.txt          # pip dependencies
├── requirements_conda.txt        # conda dependencies
└── .pre-commit-config.yaml       # code formatting hooks
```

## Setup

### With pip

```bash
pip install -r requirements_pip.txt
```

### With conda

```bash
conda install --file requirements_conda.txt
```

## Running the Examples

Each example computes the first and second derivatives of the Neo-Hookean strain energy density using the respective AD library, compares them to the analytical solution, and reports errors and runtimes.

```bash
# num-dual (dual numbers / forward-mode AD)
cd python/num-dual
python neo-hooke-example.py

# autograd (reverse-mode AD)
cd python/autograd
python neo-hooke-example.py

# JAX (JIT-compiled reverse-mode AD)
cd python/jax
python neo-hooke-example.py

# PyTorch (reverse-mode AD)
cd python/pytorch
python neo-hooke-example.py
```

## Neo-Hookean Example

The example evaluates the strain energy density

$$\Psi(\mathbf{C}) = \frac{K}{8}(J - J^{-1})^2 + \frac{G}{2}(\mathrm{tr}\,\mathbf{C}\, J^{-2/3} - 3)$$

where $J = \sqrt{\det \mathbf{C}}$ is the volume ratio, $K$ is the bulk modulus, and $G$ is the shear modulus.
The first Piola-Kirchhoff stress $\partial\Psi/\partial\mathbf{C}$ and the material tangent $\partial^2\Psi/\partial\mathbf{C}^2$ are computed and compared against analytical expressions.

## Code Quality

The repository uses [pre-commit](https://pre-commit.com/) hooks for automated code formatting. Install them with:

```bash
pip install pre-commit
pre-commit install
```

The hooks enforce consistent formatting via `black`, `isort`, `autoflake`, `flake8`, and `clang-format`.
