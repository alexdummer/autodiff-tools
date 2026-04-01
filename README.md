# autodiff-tools

Comparison of Python automatic differentiation (AD) libraries for computing first and second derivatives of scalar functions with respect to array inputs, demonstrated on a Neo-Hookean hyperelastic strain energy density.

## Tools

### Python

| Tool | Approach | Package |
|---|---|---|
| [autograd](https://github.com/HIPS/autograd) | Operator overloading (reverse-mode) | `autograd` |
| [JAX](https://jax.readthedocs.io) | JIT-compiled reverse-mode AD | `jax` |
| [PyTorch](https://pytorch.org) | Operator overloading (reverse-mode) | `torch` |
| [num-dual](https://github.com/itt-ustutt/num-dual) | (Hyper-)dual numbers (forward-mode) | `num-dual` |

Each Python tool is wrapped behind a common interface exposing `df_dT(f, T)` and `d2f_dT2(f, T)`. Results are validated against analytical derivatives in `python/utils/neohooke.py`.

### C++

| Tool | Approach | Library |
|---|---|---|
| [autodiff](https://autodiff.github.io) | Dual numbers (forward-mode) | `autodiff` |

The C++ example uses the header-only `autodiff` library and exposes the same `df_dT` / `d2f_dT2` interface in `cpp/autodiff/interface.h`. The analytical reference is implemented in `cpp/utils/neohooke.h`.

## Setup

### Python

```bash
pip install -r requirements_pip.txt       # pip
conda install --file requirements_conda.txt  # conda
```

### C++

Install the [Eigen3](https://eigen.tuxfamily.org) dependency and the header-only [autodiff](https://autodiff.github.io) library:

```bash
# Ubuntu / Debian
sudo apt-get install -y libeigen3-dev

# macOS (Homebrew)
brew install eigen

# autodiff (header-only) – copy headers to a system include path, e.g.:
git clone --depth 1 https://github.com/autodiff/autodiff.git
sudo cp -r autodiff/autodiff /usr/local/include/
```

## Usage

### Python

Run the unified example from the `python/` directory, selecting the desired backend:

```bash
cd python
python neo-hooke-example.py --backend autograd   # operator-overloading reverse-mode AD
python neo-hooke-example.py --backend jax        # JIT-compiled reverse-mode AD
python neo-hooke-example.py --backend num-dual   # forward-mode AD via dual numbers
python neo-hooke-example.py --backend pytorch    # operator-overloading reverse-mode AD
```

Each run reports the derivative errors relative to the analytical solution and the runtime of both approaches.

### C++

Build and run the example from the `cpp/autodiff/` directory:

```bash
cd cpp/autodiff
bash build_and_run_example.sh
```
