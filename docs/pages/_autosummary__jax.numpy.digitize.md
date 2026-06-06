- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.digitize

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.digitize.rst "Download source file")
-  .pdf

# jax.numpy.digitize

## Contents

- [`digitize()`](#jax.numpy.digitize)

# jax.numpy.digitize[\#](#jax-numpy-digitize "Link to this heading")

jax.numpy.digitize(*x*, *bins*, *right=False*, *\**, *method=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L9403-L9456)[\#](#jax.numpy.digitize "Link to this definition")  
Convert an array to bin indices.

JAX implementation of [`numpy.digitize()`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html#numpy.digitize "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – array of values to digitize.

- **bins** (*ArrayLike*) – 1D array of bin edges. Must be monotonically increasing or decreasing.

- **right** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if true, the intervals include the right bin edges. If false (default) the intervals include the left bin edges.

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – optional method argument to be passed to [`searchsorted()`](jax.numpy.searchsorted.html#jax.numpy.searchsorted "jax.numpy.searchsorted"). See that function for available options.

Returns:  
An integer array of the same shape as `x` indicating the bin number that the values are in.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.searchsorted()`](jax.numpy.searchsorted.html#jax.numpy.searchsorted "jax.numpy.searchsorted"): find insertion indices for values in a sorted array.

- [`jax.numpy.histogram()`](jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram"): compute frequency of array values within specified bins.

Examples

    >>> x = jnp.array([1.0, 2.0, 2.5, 1.5, 3.0, 3.5])
    >>> bins = jnp.array([1, 2, 3])
    >>> jnp.digitize(x, bins)
    Array([1, 2, 2, 1, 3, 3], dtype=int32)
    >>> jnp.digitize(x, bins, right=True)
    Array([0, 1, 2, 1, 2, 3], dtype=int32)

`digitize` supports reverse-ordered bins as well:

    >>> bins = jnp.array([3, 2, 1])
    >>> jnp.digitize(x, bins)
    Array([2, 1, 1, 2, 0, 0], dtype=int32)

[](jax.numpy.diff.html "previous page")

previous

jax.numpy.diff

[](jax.numpy.divide.html "next page")

next

jax.numpy.divide

Contents

- [`digitize()`](#jax.numpy.digitize)

By The JAX authors

© Copyright 2024, The JAX Authors.\
