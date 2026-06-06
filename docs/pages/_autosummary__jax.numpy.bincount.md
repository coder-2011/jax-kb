- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.bincount

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.bincount.rst "Download source file")
-  .pdf

# jax.numpy.bincount

## Contents

- [`bincount()`](#jax.numpy.bincount)

# jax.numpy.bincount[\#](#jax-numpy-bincount "Link to this heading")

jax.numpy.bincount(*x*, *weights=None*, *minlength=0*, *\**, *length=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2885-L2991)[\#](#jax.numpy.bincount "Link to this definition")  
Count the number of occurrences of each value in an integer array.

JAX implementation of [`numpy.bincount()`](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html#numpy.bincount "(in NumPy v2.4)").

For an array of non-negative integers `x`, this function returns an array `counts` of size `x.max()`` ``+`` ``1`, such that `counts[i]` contains the number of occurrences of the value `i` in `x`.

The JAX version has a few differences from the NumPy version:

- In NumPy, passing an array `x` with negative entries will result in an error. In JAX, negative values are clipped to zero.

- JAX adds an optional `length` parameter which can be used to statically specify the length of the output array so that this function can be used with transformations like [`jax.jit()`](jax.jit.html#jax.jit "jax.jit"). In this case, items larger than length + 1 will be dropped.

Parameters:  
- **x** (*ArrayLike*) – 1-dimensional array of non-negative integers

- **weights** (*ArrayLike* *\|* *None*) – optional array of weights associated with `x`. If not specified, the weight for each entry will be `1`.

- **minlength** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the minimum length of the output counts array.

- **length** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – the length of the output counts array. Must be specified statically for `bincount` to be used with [`jax.jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – (optional) `NamedSharding` or `P` to which the created array will be committed. Use out_sharding argument, if using explicit sharding ([https://docs.jax.dev/en/latest/parallel.html](https://docs.jax.dev/en/latest/parallel.html)).

Returns:  
An array of counts or summed weights reflecting the number of occurrences of values in `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.histogram()`](jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram")

- [`jax.numpy.digitize()`](jax.numpy.digitize.html#jax.numpy.digitize "jax.numpy.digitize")

- [`jax.numpy.unique_counts()`](jax.numpy.unique_counts.html#jax.numpy.unique_counts "jax.numpy.unique_counts")

Examples

Basic bincount:

    >>> x = jnp.array([1, 1, 2, 3, 3, 3])
    >>> jnp.bincount(x)
    Array([0, 2, 1, 3], dtype=int32)

Weighted bincount:

    >>> weights = jnp.array([1, 2, 3, 4, 5, 6])
    >>> jnp.bincount(x, weights)
    Array([ 0,  3,  3, 15], dtype=int32)

Specifying a static `length` makes this jit-compatible:

    >>> jit_bincount = jax.jit(jnp.bincount, static_argnames=['length'])
    >>> jit_bincount(x, length=5)
    Array([0, 2, 1, 3, 0], dtype=int32)

Any negative numbers are clipped to the first bin, and numbers beyond the specified `length` are dropped:

    >>> x = jnp.array([-1, -1, 1, 3, 10])
    >>> jnp.bincount(x, length=5)
    Array([2, 1, 0, 1, 0], dtype=int32)

[](jax.numpy.bartlett.html "previous page")

previous

jax.numpy.bartlett

[](jax.numpy.bitwise_and.html "next page")

next

jax.numpy.bitwise_and

Contents

- [`bincount()`](#jax.numpy.bincount)

By The JAX authors

© Copyright 2024, The JAX Authors.\
