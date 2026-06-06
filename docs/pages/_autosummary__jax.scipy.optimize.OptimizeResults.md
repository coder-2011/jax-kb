- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.optimize.OptimizeResults

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.optimize.OptimizeResults.rst "Download source file")
-  .pdf

# jax.scipy.optimize.OptimizeResults

## Contents

- [`OptimizeResults`](#jax.scipy.optimize.OptimizeResults)
  - [`OptimizeResults.__init__()`](#jax.scipy.optimize.OptimizeResults.__init__)

# jax.scipy.optimize.OptimizeResults[\#](#jax-scipy-optimize-optimizeresults "Link to this heading")

*class* jax.scipy.optimize.OptimizeResults(*x*, *success*, *status*, *fun*, *jac*, *hess_inv*, *nfev*, *njev*, *nit*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/optimize/minimize.py#L26-L51)[\#](#jax.scipy.optimize.OptimizeResults "Link to this definition")  
Object holding optimization results.

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – final solution.

- **success** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – `True` if optimization succeeded.

- **status** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – integer solver specific return code. 0 means converged (nominal), 1=max BFGS iters reached, 3=zoom failed, 4=saddle point reached, 5=max line search iters reached, -1=undefined

- **fun** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – final function value.

- **jac** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – final jacobian array.

- **hess_inv** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*) – final inverse Hessian estimate.

- **nfev** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – integer number of function calls used.

- **njev** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – integer number of gradient evaluations.

- **nit** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array")) – integer number of iterations of the optimization algorithm.

\_\_init\_\_()[\#](#jax.scipy.optimize.OptimizeResults.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.scipy.optimize.OptimizeResults.__init__ "jax.scipy.optimize.OptimizeResults.__init__")() |  |
| `count`(value, /) | Return number of occurrences of value. |
| `index`(value\[, start, stop\]) | Return first index of value. |

Attributes

|            |                          |
|------------|--------------------------|
| `fun`      | Alias for field number 3 |
| `hess_inv` | Alias for field number 5 |
| `jac`      | Alias for field number 4 |
| `nfev`     | Alias for field number 6 |
| `nit`      | Alias for field number 8 |
| `njev`     | Alias for field number 7 |
| `status`   | Alias for field number 2 |
| `success`  | Alias for field number 1 |
| `x`        | Alias for field number 0 |

[](jax.scipy.optimize.minimize.html "previous page")

previous

jax.scipy.optimize.minimize

[](jax.scipy.signal.fftconvolve.html "next page")

next

jax.scipy.signal.fftconvolve

Contents

- [`OptimizeResults`](#jax.scipy.optimize.OptimizeResults)
  - [`OptimizeResults.__init__()`](#jax.scipy.optimize.OptimizeResults.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
