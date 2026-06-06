- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.cbrt

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.cbrt.rst "Download source file")
-  .pdf

# jax.numpy.cbrt

## Contents

- [`cbrt()`](#jax.numpy.cbrt)

# jax.numpy.cbrt[\#](#jax-numpy-cbrt "Link to this heading")

jax.numpy.cbrt(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1170-L1197)[\#](#jax.numpy.cbrt "Link to this definition")  
Calculates element-wise cube root of the input array.

JAX implementation of [`numpy.cbrt`](https://numpy.org/doc/stable/reference/generated/numpy.cbrt.html#numpy.cbrt "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar. `complex` dtypes are not supported.

Returns:  
An array containing the cube root of the elements of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.sqrt()`](jax.numpy.sqrt.html#jax.numpy.sqrt "jax.numpy.sqrt"): Calculates the element-wise non-negative square root of the input.

- [`jax.numpy.square()`](jax.numpy.square.html#jax.numpy.square "jax.numpy.square"): Calculates the element-wise square of the input.

Examples

    >>> x = jnp.array([[216, 125, 64],
    ...                [-27, -8, -1]])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.cbrt(x)
    Array([[ 6.,  5.,  4.],
           [-3., -2., -1.]], dtype=float32)

[](jax.numpy.can_cast.html "previous page")

previous

jax.numpy.can_cast

[](jax.numpy.cdouble.html "next page")

next

jax.numpy.cdouble

Contents

- [`cbrt()`](#jax.numpy.cbrt)

By The JAX authors

© Copyright 2024, The JAX Authors.\
