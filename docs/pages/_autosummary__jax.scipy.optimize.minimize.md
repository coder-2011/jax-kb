- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.optimize.minimize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.optimize.minimize.rst "Download source file")
-  .pdf

# jax.scipy.optimize.minimize

## Contents

- [`minimize()`](#jax.scipy.optimize.minimize)

# jax.scipy.optimize.minimize[\#](#jax-scipy-optimize-minimize "Link to this heading")

jax.scipy.optimize.minimize(*fun*, *x0*, *args=()*, *\**, *method*, *tol=None*, *options=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/optimize/minimize.py#L53-L134)[\#](#jax.scipy.optimize.minimize "Link to this definition")  
Minimization of scalar function of one or more variables.

This API for this function matches SciPy with some minor deviations:

- Gradients of `fun` are calculated automatically using JAX’s autodiff support when required.

- The `method` argument is required. You must specify a solver.

- Various optional arguments in the SciPy interface have not yet been implemented.

- Optimization results may differ from SciPy due to differences in the line search implementation.

`minimize` supports [`jit()`](jax.jit.html#jax.jit "jax.jit") compilation. It does not yet support differentiation or arguments in the form of multi-dimensional arrays, but support for both is planned.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – the objective function to be minimized, `fun(x,`` ``*args)`` ``->`` ``float`, where `x` is a 1-D array with shape `(n,)` and `args` is a tuple of the fixed parameters needed to completely specify the function. `fun` must support differentiation.

- **x0** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – initial guess. Array of real elements of size `(n,)`, where `n` is the number of independent variables.

- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")) – extra arguments passed to the objective function.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – solver type. Currently only `"BFGS"` is supported.

- **tol** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – tolerance for termination. For detailed control, use solver-specific options.

- **options** (*Mapping\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*,* *Any\]* *\|* *None*) –

  a dictionary of solver options. All methods accept the following generic options:

  - maxiter (int): Maximum number of iterations to perform. Depending on the method each iteration may use several function evaluations.

Returns:  
An [`OptimizeResults`](jax.scipy.optimize.OptimizeResults.html#jax.scipy.optimize.OptimizeResults "jax.scipy.optimize.OptimizeResults") object.

Return type:  
[OptimizeResults](jax.scipy.optimize.OptimizeResults.html#jax.scipy.optimize.OptimizeResults "jax.scipy.optimize.OptimizeResults")

[](jax.scipy.ndimage.map_coordinates.html "previous page")

previous

jax.scipy.ndimage.map_coordinates

[](jax.scipy.optimize.OptimizeResults.html "next page")

next

jax.scipy.optimize.OptimizeResults

Contents

- [`minimize()`](#jax.scipy.optimize.minimize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
