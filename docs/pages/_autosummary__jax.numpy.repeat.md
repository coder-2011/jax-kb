- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.repeat

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.repeat.rst "Download source file")
-  .pdf

# jax.numpy.repeat

## Contents

- [`repeat()`](#jax.numpy.repeat)

# jax.numpy.repeat[\#](#jax-numpy-repeat "Link to this heading")

jax.numpy.repeat(*a*, *repeats*, *axis=None*, *\**, *total_repeat_length=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6259-L6356)[\#](#jax.numpy.repeat "Link to this definition")  
Construct an array from repeated elements.

JAX implementation of [`numpy.repeat()`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – N-dimensional array

- **repeats** (*ArrayLike*) – 1D integer array specifying the number of repeats. Must match the length of the repeated axis.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer specifying the axis of `a` along which to construct the repeated array. If None (default) then `a` is first flattened.

- **total_repeat_length** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – this must be specified statically for `jnp.repeat` to be compatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations. If `sum(repeats)` is larger than the specified `total_repeat_length`, the remaining values will be discarded. If `sum(repeats)` is smaller than `total_repeat_length`, the final value will be repeated.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
an array constructed from repeated values of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.tile()`](jax.numpy.tile.html#jax.numpy.tile "jax.numpy.tile"): repeat a full array rather than individual values.

Examples

Repeat each value twice along the last axis:

    >>> a = jnp.array([[1, 2],
    ...                [3, 4]])
    >>> jnp.repeat(a, 2, axis=-1)
    Array([[1, 1, 2, 2],
           [3, 3, 4, 4]], dtype=int32)

If `axis` is not specified, the input array will be flattened:

    >>> jnp.repeat(a, 2)
    Array([1, 1, 2, 2, 3, 3, 4, 4], dtype=int32)

Pass an array to `repeats` to repeat each value a different number of times:

    >>> repeats = jnp.array([2, 3])
    >>> jnp.repeat(a, repeats, axis=1)
    Array([[1, 1, 2, 2, 2],
           [3, 3, 4, 4, 4]], dtype=int32)

In order to use `repeat` within `jit` and other JAX transformations, the size of the output must be specified statically using `total_repeat_length`:

    >>> jit_repeat = jax.jit(jnp.repeat, static_argnames=['axis', 'total_repeat_length'])
    >>> jit_repeat(a, repeats, axis=1, total_repeat_length=5)
    Array([[1, 1, 2, 2, 2],
           [3, 3, 4, 4, 4]], dtype=int32)

If total_repeat_length is smaller than `sum(repeats)`, the result will be truncated:

    >>> jit_repeat(a, repeats, axis=1, total_repeat_length=4)
    Array([[1, 1, 2, 2],
           [3, 3, 4, 4]], dtype=int32)

If it is larger, then the additional entries will be filled with the final value:

    >>> jit_repeat(a, repeats, axis=1, total_repeat_length=7)
    Array([[1, 1, 2, 2, 2, 2, 2],
           [3, 3, 4, 4, 4, 4, 4]], dtype=int32)

[](jax.numpy.remainder.html "previous page")

previous

jax.numpy.remainder

[](jax.numpy.reshape.html "next page")

next

jax.numpy.reshape

Contents

- [`repeat()`](#jax.numpy.repeat)

By The JAX authors

© Copyright 2024, The JAX Authors.\
