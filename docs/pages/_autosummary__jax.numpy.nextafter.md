- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nextafter

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nextafter.rst "Download source file")
-  .pdf

# jax.numpy.nextafter

## Contents

- [`nextafter()`](#jax.numpy.nextafter)

# jax.numpy.nextafter[\#](#jax-numpy-nextafter "Link to this heading")

jax.numpy.nextafter(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1798-L1824)[\#](#jax.numpy.nextafter "Link to this definition")  
Return element-wise next floating point value after `x` towards `y`.

JAX implementation of [`numpy.nextafter`](https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html#numpy.nextafter "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – scalar or array. Specifies the value after which the next number is found.

- **y** (*ArrayLike*) – scalar or array. Specifies the direction towards which the next number is found. `x` and `y` should either have same shape or be broadcast compatible.

Returns:  
An array containing the next representable number of `x` in the direction of `y`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jnp.nextafter(2, 1)  
    Array(1.9999999, dtype=float32, weak_type=True)
    >>> x = jnp.array([3, -2, 1])
    >>> y = jnp.array([2, -1, 2])
    >>> jnp.nextafter(x, y)  
    Array([ 2.9999998, -1.9999999,  1.0000001], dtype=float32)

[](jax.numpy.negative.html "previous page")

previous

jax.numpy.negative

[](jax.numpy.nonzero.html "next page")

next

jax.numpy.nonzero

Contents

- [`nextafter()`](#jax.numpy.nextafter)

By The JAX authors

© Copyright 2024, The JAX Authors.\
