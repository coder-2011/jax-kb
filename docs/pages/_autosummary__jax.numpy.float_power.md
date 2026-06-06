- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.float_power

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.float_power.rst "Download source file")
-  .pdf

# jax.numpy.float_power

## Contents

- [`float_power()`](#jax.numpy.float_power)

# jax.numpy.float_power[\#](#jax-numpy-float-power "Link to this heading")

jax.numpy.float_power(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1749-L1796)[\#](#jax.numpy.float_power "Link to this definition")  
Calculate element-wise base `x` exponential of `y`.

JAX implementation of [`numpy.float_power`](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html#numpy.float_power "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – scalar or array. Specifies the bases.

- **y** (*ArrayLike*) – scalar or array. Specifies the exponents. `x` and `y` should either have same shape or be broadcast compatible.

Returns:  
An array containing the base `x` exponentials of `y`, promoting to the inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.exp()`](jax.numpy.exp.html#jax.numpy.exp "jax.numpy.exp"): Calculates element-wise exponential of the input.

- [`jax.numpy.exp2()`](jax.numpy.exp2.html#jax.numpy.exp2 "jax.numpy.exp2"): Calculates base-2 exponential of each element of the input.

Examples

Inputs with same shape:

    >>> x = jnp.array([3, 1, -5])
    >>> y = jnp.array([2, 4, -1])
    >>> jnp.float_power(x, y)
    Array([ 9. ,  1. , -0.2], dtype=float32)

Inputs with broadcast compatibility:

    >>> x1 = jnp.array([[2, -4, 1],
    ...                 [-1, 2, 3]])
    >>> y1 = jnp.array([-2, 1, 4])
    >>> jnp.float_power(x1, y1)
    Array([[ 0.25, -4.  ,  1.  ],
           [ 1.  ,  2.  , 81.  ]], dtype=float32)

`jnp.float_power` produces `nan` for negative values raised to a non-integer values.

    >>> jnp.float_power(-3, 1.7)
    Array(nan, dtype=float32, weak_type=True)

[](jax.numpy.float_.html "previous page")

previous

jax.numpy.float\_

[](jax.numpy.float16.html "next page")

next

jax.numpy.float16

Contents

- [`float_power()`](#jax.numpy.float_power)

By The JAX authors

© Copyright 2024, The JAX Authors.\
