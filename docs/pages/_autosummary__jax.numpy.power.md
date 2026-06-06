- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.power

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.power.rst "Download source file")
-  .pdf

# jax.numpy.power

## Contents

- [`power()`](#jax.numpy.power)

# jax.numpy.power[\#](#jax-numpy-power "Link to this heading")

jax.numpy.power(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2602-L2696)[\#](#jax.numpy.power "Link to this definition")  
Calculate element-wise base `x1` exponential of `x2`.

JAX implementation of [`numpy.power`](https://numpy.org/doc/stable/reference/generated/numpy.power.html#numpy.power "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – scalar or array. Specifies the bases.

- **x2** (*ArrayLike*) – scalar or array. Specifies the exponent. `x1` and `x2` should either have same shape or be broadcast compatible.

Returns:  
An array containing the base `x1` exponentials of `x2` with same dtype as input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

- When `x2` is a concrete integer scalar, `jnp.power` lowers to [`jax.lax.integer_pow()`](jax.lax.integer_pow.html#jax.lax.integer_pow "jax.lax.integer_pow").

- When `x2` is a traced scalar or an array, `jnp.power` lowers to [`jax.lax.pow()`](jax.lax.pow.html#jax.lax.pow "jax.lax.pow").

- `jnp.power` raises a `TypeError` for integer type raised to a concrete negative integer power. For a non-concrete power, the operation is invalid and the returned value is implementation-defined.

- `jnp.power` returns `nan` for negative value raised to the power of non-integer values.

See also

- [`jax.lax.pow()`](jax.lax.pow.html#jax.lax.pow "jax.lax.pow"): Computes element-wise power, \\x^y\\.

- [`jax.lax.integer_pow()`](jax.lax.integer_pow.html#jax.lax.integer_pow "jax.lax.integer_pow"): Computes element-wise power \\x^y\\, where \\y\\ is a fixed integer.

- [`jax.numpy.float_power()`](jax.numpy.float_power.html#jax.numpy.float_power "jax.numpy.float_power"): Computes the first array raised to the power of second array, element-wise, by promoting to the inexact dtype.

- [`jax.numpy.pow()`](jax.numpy.pow.html#jax.numpy.pow "jax.numpy.pow"): Computes the first array raised to the power of second array, element-wise.

Examples

Inputs with scalar integers:

    >>> jnp.power(4, 3)
    Array(64, dtype=int32, weak_type=True)

Inputs with same shape:

    >>> x1 = jnp.array([2, 4, 5])
    >>> x2 = jnp.array([3, 0.5, 2])
    >>> jnp.power(x1, x2)
    Array([ 8.,  2., 25.], dtype=float32)

Inputs with broadcast compatibility:

    >>> x3 = jnp.array([-2, 3, 1])
    >>> x4 = jnp.array([[4, 1, 6],
    ...                 [1.3, 3, 5]])
    >>> jnp.power(x3, x4)
    Array([[16.,  3.,  1.],
           [nan, 27.,  1.]], dtype=float32)

[](jax.numpy.pow.html "previous page")

previous

jax.numpy.pow

[](jax.numpy.printoptions.html "next page")

next

jax.numpy.printoptions

Contents

- [`power()`](#jax.numpy.power)

By The JAX authors

© Copyright 2024, The JAX Authors.\
