- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.true_divide

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.true_divide.rst "Download source file")
-  .pdf

# jax.numpy.true_divide

## Contents

- [`true_divide()`](#jax.numpy.true_divide)

# jax.numpy.true_divide[\#](#jax-numpy-true-divide "Link to this heading")

jax.numpy.true_divide(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2441-L2481)[\#](#jax.numpy.true_divide "Link to this definition")  
Calculates the division of x1 by x2 element-wise

JAX implementation of `numpy.true_divide()`.

Parameters:  
- **x1** (*ArrayLike*) – Input array, the dividend

- **x2** (*ArrayLike*) – Input array, the divisor

Returns:  
An array containing the elementwise quotients, will always use floating point division.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x1 = jnp.array([3, 4, 5])
    >>> x2 = 2
    >>> jnp.true_divide(x1, x2)
    Array([1.5, 2. , 2.5], dtype=float32)

    >>> x1 = 24
    >>> x2 = jnp.array([3, 4, 6j])
    >>> jnp.true_divide(x1, x2)
    Array([8.+0.j, 6.+0.j, 0.-4.j], dtype=complex64)

    >>> x1 = jnp.array([1j, 9+5j, -4+2j])
    >>> x2 = 3j
    >>> jnp.true_divide(x1, x2)
    Array([0.33333334+0.j       , 1.6666666 -3.j       ,
           0.6666667 +1.3333334j], dtype=complex64)

See also

[`jax.numpy.floor_divide()`](jax.numpy.floor_divide.html#jax.numpy.floor_divide "jax.numpy.floor_divide") for integer division

[](jax.numpy.triu_indices_from.html "previous page")

previous

jax.numpy.triu_indices_from

[](jax.numpy.trunc.html "next page")

next

jax.numpy.trunc

Contents

- [`true_divide()`](#jax.numpy.true_divide)

By The JAX authors

© Copyright 2024, The JAX Authors.\
