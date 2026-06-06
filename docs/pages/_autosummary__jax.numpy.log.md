- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.log

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.log.rst "Download source file")
-  .pdf

# jax.numpy.log

## Contents

- [`log()`](#jax.numpy.log)

# jax.numpy.log[\#](#jax-numpy-log "Link to this heading")

jax.numpy.log(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L456-L494)[\#](#jax.numpy.log "Link to this definition")  
Calculate element-wise natural logarithm of the input.

JAX implementation of [`numpy.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html#numpy.log "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the logarithm of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.exp()`](jax.numpy.exp.html#jax.numpy.exp "jax.numpy.exp"): Calculates element-wise exponential of the input.

- [`jax.numpy.log2()`](jax.numpy.log2.html#jax.numpy.log2 "jax.numpy.log2"): Calculates base-2 logarithm of each element of input.

- [`jax.numpy.log1p()`](jax.numpy.log1p.html#jax.numpy.log1p "jax.numpy.log1p"): Calculates element-wise logarithm of one plus input.

Examples

`jnp.log` and `jnp.exp` are inverse functions of each other. Applying `jnp.log` on the result of `jnp.exp(x)` yields the original input `x`.

    >>> x = jnp.array([2, 3, 4, 5])
    >>> jnp.log(jnp.exp(x))
    Array([2., 3., 4., 5.], dtype=float32)

Using `jnp.log` we can demonstrate well-known properties of logarithms, such as \\log(a\*b) = log(a)+log(b)\\.

    >>> x1 = jnp.array([2, 1, 3, 1])
    >>> x2 = jnp.array([1, 3, 2, 4])
    >>> jnp.allclose(jnp.log(x1*x2), jnp.log(x1)+jnp.log(x2))
    Array(True, dtype=bool)

[](jax.numpy.load.html "previous page")

previous

jax.numpy.load

[](jax.numpy.log10.html "next page")

next

jax.numpy.log10

Contents

- [`log()`](#jax.numpy.log)

By The JAX authors

© Copyright 2024, The JAX Authors.\
