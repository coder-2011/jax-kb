- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.log1p

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.log1p.rst "Download source file")
-  .pdf

# jax.numpy.log1p

## Contents

- [`log1p()`](#jax.numpy.log1p)

# jax.numpy.log1p[\#](#jax-numpy-log1p "Link to this heading")

jax.numpy.log1p(*x*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L541-L582)[\#](#jax.numpy.log1p "Link to this definition")  
Calculates element-wise logarithm of one plus input, `log(x+1)`.

JAX implementation of [`numpy.log1p`](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html#numpy.log1p "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array or scalar.

Returns:  
An array containing the logarithm of one plus of each element in `x`, promotes to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.log1p` is more accurate than when using the naive computation of `log(x+1)` for small values of `x`.

See also

- [`jax.numpy.expm1()`](jax.numpy.expm1.html#jax.numpy.expm1 "jax.numpy.expm1"): Calculates \\e^x-1\\ of each element of the input.

- [`jax.numpy.log2()`](jax.numpy.log2.html#jax.numpy.log2 "jax.numpy.log2"): Calculates base-2 logarithm of each element of input.

- [`jax.numpy.log()`](jax.numpy.log.html#jax.numpy.log "jax.numpy.log"): Calculates element-wise logarithm of the input.

Examples

    >>> x = jnp.array([2, 5, 9, 4])
    >>> jnp.allclose(jnp.log1p(x), jnp.log(x+1))
    Array(True, dtype=bool)

For values very close to 0, `jnp.log1p(x)` is more accurate than `jnp.log(x+1)`:

    >>> x1 = jnp.array([1e-4, 1e-6, 2e-10])
    >>> jnp.expm1(jnp.log1p(x1))  
    Array([1.00000005e-04, 9.99999997e-07, 2.00000003e-10], dtype=float32)
    >>> jnp.expm1(jnp.log(x1+1))  
    Array([1.000166e-04, 9.536743e-07, 0.000000e+00], dtype=float32)

[](jax.numpy.log10.html "previous page")

previous

jax.numpy.log10

[](jax.numpy.log2.html "next page")

next

jax.numpy.log2

Contents

- [`log1p()`](#jax.numpy.log1p)

By The JAX authors

© Copyright 2024, The JAX Authors.\
