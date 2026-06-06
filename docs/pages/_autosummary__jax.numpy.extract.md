- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.extract

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.extract.rst "Download source file")
-  .pdf

# jax.numpy.extract

## Contents

- [`extract()`](#jax.numpy.extract)

# jax.numpy.extract[\#](#jax-numpy-extract "Link to this heading")

jax.numpy.extract(*condition*, *arr*, *\**, *size=None*, *fill_value=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8897-L8957)[\#](#jax.numpy.extract "Link to this definition")  
Return the elements of an array that satisfy a condition.

JAX implementation of [`numpy.extract()`](https://numpy.org/doc/stable/reference/generated/numpy.extract.html#numpy.extract "(in NumPy v2.4)").

Parameters:  
- **condition** (*ArrayLike*) – array of conditions. Will be converted to boolean and flattened to 1D.

- **arr** (*ArrayLike*) – array of values to extract. Will be flattened to 1D.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional static size for output. Must be specified in order for `extract` to be compatible with JAX transformations like [`jit()`](jax.jit.html#jax.jit "jax.jit") or [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap").

- **fill_value** (*ArrayLike*) – if `size` is specified, fill padded entries with this value (default: 0).

Returns:  
1D array of extracted entries . If `size` is specified, the result will have shape `(size,)` and be right-padded with `fill_value`. If `size` is not specified, the output shape will depend on the number of True entries in `condition`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

This function does not require strict shape agreement between `condition` and `arr`. If `condition.size`` ``>`` ``arr.size`, then `condition` will be truncated, and if `arr.size`` ``>`` ``condition.size`, then `arr` will be truncated.

See also

[`jax.numpy.compress()`](jax.numpy.compress.html#jax.numpy.compress "jax.numpy.compress"): multi-dimensional version of `extract`.

Examples

Extract values from a 1D array:

    >>> x = jnp.array([1, 2, 3, 4, 5, 6])
    >>> mask = (x % 2 == 0)
    >>> jnp.extract(mask, x)
    Array([2, 4, 6], dtype=int32)

In the simplest case, this is equivalent to boolean indexing:

    >>> x[mask]
    Array([2, 4, 6], dtype=int32)

For use with JAX transformations, you can pass the `size` argument to specify a static shape for the output, along with an optional `fill_value` that defaults to zero:

    >>> jnp.extract(mask, x, size=len(x), fill_value=0)
    Array([2, 4, 6, 0, 0, 0], dtype=int32)

Notice that unlike with boolean indexing, `extract` does not require strict agreement between the sizes of the array and condition, and will effectively truncate both to the minimum size:

    >>> short_mask = jnp.array([False, True])
    >>> jnp.extract(short_mask, x)
    Array([2], dtype=int32)
    >>> long_mask = jnp.array([True, False, True, False, False, False, False, False])
    >>> jnp.extract(long_mask, x)
    Array([1, 3], dtype=int32)

[](jax.numpy.expm1.html "previous page")

previous

jax.numpy.expm1

[](jax.numpy.eye.html "next page")

next

jax.numpy.eye

Contents

- [`extract()`](#jax.numpy.extract)

By The JAX authors

© Copyright 2024, The JAX Authors.\
