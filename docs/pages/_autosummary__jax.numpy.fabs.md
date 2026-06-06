- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fabs

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fabs.rst "Download source file")
-  .pdf

# jax.numpy.fabs

## Contents

- [`fabs()`](#jax.numpy.fabs)

# jax.numpy.fabs[\#](#jax-numpy-fabs "Link to this heading")

jax.numpy.fabs(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L84-L127)[\#](#jax.numpy.fabs "Link to this definition")  
Compute the element-wise absolute values of the real-valued input.

JAX implementation of [`numpy.fabs`](https://numpy.org/doc/stable/reference/generated/numpy.fabs.html#numpy.fabs "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar. Must not have a complex dtype.

Returns:  
An array with same shape as `x` and dtype float, containing the element-wise absolute values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.absolute()`](jax.numpy.absolute.html#jax.numpy.absolute "jax.numpy.absolute"): Computes the absolute values of the input including complex dtypes.

- [`jax.numpy.abs()`](jax.numpy.abs.html#jax.numpy.abs "jax.numpy.abs"): Computes the absolute values of the input including complex dtypes.

Examples

For integer inputs:

    >>> x = jnp.array([-5, -9, 1, 10, 15])
    >>> jnp.fabs(x)
    Array([ 5.,  9.,  1., 10., 15.], dtype=float32)

For float type inputs:

    >>> x1 = jnp.array([-1.342, 5.649, 3.927])
    >>> jnp.fabs(x1)
    Array([1.342, 5.649, 3.927], dtype=float32)

For boolean inputs:

    >>> x2 = jnp.array([True, False])
    >>> jnp.fabs(x2)
    Array([1., 0.], dtype=float32)

[](jax.numpy.eye.html "previous page")

previous

jax.numpy.eye

[](jax.numpy.fill_diagonal.html "next page")

next

jax.numpy.fill_diagonal

Contents

- [`fabs()`](#jax.numpy.fabs)

By The JAX authors

© Copyright 2024, The JAX Authors.\
