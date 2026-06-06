- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ldexp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ldexp.rst "Download source file")
-  .pdf

# jax.numpy.ldexp

## Contents

- [`ldexp()`](#jax.numpy.ldexp)

# jax.numpy.ldexp[\#](#jax-numpy-ldexp "Link to this heading")

jax.numpy.ldexp(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2966-L3023)[\#](#jax.numpy.ldexp "Link to this definition")  
Compute x1 \* 2 \*\* x2

JAX implementation of `numpy.ldexp()`.

Note that XLA does not provide an `ldexp` operation, so this is implemneted in JAX via a standard multiplication and exponentiation.

Parameters:  
- **x1** (*ArrayLike*) – real-valued input array.

- **x2** (*ArrayLike*) – integer input array. Must be broadcast-compatible with `x1`.

Returns:  
`x1`` ``*`` ``2`` ``**`` ``x2` computed element-wise.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.frexp()`](jax.numpy.frexp.html#jax.numpy.frexp "jax.numpy.frexp"): decompose values into mantissa and exponent.

Examples

    >>> x1 = jnp.arange(5.0)
    >>> x2 = 10
    >>> jnp.ldexp(x1, x2)
    Array([   0., 1024., 2048., 3072., 4096.], dtype=float32)

`ldexp` can be used to reconstruct the input to `frexp`:

    >>> x = jnp.array([2., 3., 5., 11.])
    >>> m, e = jnp.frexp(x)
    >>> m
    Array([0.5   , 0.75  , 0.625 , 0.6875], dtype=float32)
    >>> e
    Array([2, 2, 3, 4], dtype=int32)
    >>> jnp.ldexp(m, e)
    Array([ 2.,  3.,  5., 11.], dtype=float32)

[](jax.numpy.lcm.html "previous page")

previous

jax.numpy.lcm

[](jax.numpy.left_shift.html "next page")

next

jax.numpy.left_shift

Contents

- [`ldexp()`](#jax.numpy.ldexp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
