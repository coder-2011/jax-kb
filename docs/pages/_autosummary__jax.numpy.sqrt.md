- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.sqrt

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.sqrt.rst "Download source file")
-  .pdf

# jax.numpy.sqrt

## Contents

- [`sqrt()`](#jax.numpy.sqrt)

# jax.numpy.sqrt[\#](#jax-numpy-sqrt "Link to this heading")

jax.numpy.sqrt(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1135-L1168)[\#](#jax.numpy.sqrt "Link to this definition")  
Calculates element-wise non-negative square root of the input array.

JAX implementation of [`numpy.sqrt`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html#numpy.sqrt "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the non-negative square root of the elements of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

- For real-valued negative inputs, `jnp.sqrt` produces a `nan` output.

- For complex-valued negative inputs, `jnp.sqrt` produces a `complex` output.

See also

- [`jax.numpy.square()`](jax.numpy.square.html#jax.numpy.square "jax.numpy.square"): Calculates the element-wise square of the input.

- [`jax.numpy.power()`](jax.numpy.power.html#jax.numpy.power "jax.numpy.power"): Calculates the element-wise base `x1` exponential of `x2`.

Examples

    >>> x = jnp.array([-8-6j, 1j, 4])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.sqrt(x)
    Array([1.   -3.j   , 0.707+0.707j, 2.   +0.j   ], dtype=complex64)
    >>> jnp.sqrt(-1)
    Array(nan, dtype=float32, weak_type=True)

[](jax.numpy.split.html "previous page")

previous

jax.numpy.split

[](jax.numpy.square.html "next page")

next

jax.numpy.square

Contents

- [`sqrt()`](#jax.numpy.sqrt)

By The JAX authors

© Copyright 2024, The JAX Authors.\
