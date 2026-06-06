- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.histogram

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.histogram.rst "Download source file")
-  .pdf

# jax.numpy.histogram

## Contents

- [`histogram()`](#jax.numpy.histogram)

# jax.numpy.histogram[\#](#jax-numpy-histogram "Link to this heading")

jax.numpy.histogram(*a*, *bins=10*, *range=None*, *weights=None*, *density=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L807-L887)[\#](#jax.numpy.histogram "Link to this definition")  
Compute a 1-dimensional histogram.

JAX implementation of [`numpy.histogram()`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html#numpy.histogram "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array of values to be binned. May be any size or dimension.

- **bins** (*ArrayLike*) – Specify the number of bins in the histogram (default: 10). `bins` may also be an array specifying the locations of the bin edges.

- **range** (*Sequence\[ArrayLike\]* *\|* *None*) – tuple of scalars. Specifies the range of the data. If not specified, the range is inferred from the data.

- **weights** (*ArrayLike* *\|* *None*) – An optional array specifying the weights of the data points. Should be broadcast-compatible with `a`. If not specified, each data point is weighted equally.

- **density** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – If True, return the normalized histogram in units of counts per unit length. If False (default) return the (weighted) counts per bin.

Returns:  
A tuple of arrays `(histogram,`` ``bin_edges)`, where `histogram` contains the aggregated data, and `bin_edges` specifies the boundaries of the bins.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.bincount()`](jax.numpy.bincount.html#jax.numpy.bincount "jax.numpy.bincount"): Count the number of occurrences of each value in an array.

- [`jax.numpy.histogram2d()`](jax.numpy.histogram2d.html#jax.numpy.histogram2d "jax.numpy.histogram2d"): Compute the histogram of a 2D array.

- [`jax.numpy.histogramdd()`](jax.numpy.histogramdd.html#jax.numpy.histogramdd "jax.numpy.histogramdd"): Compute the histogram of an N-dimensional array.

- [`jax.numpy.histogram_bin_edges()`](jax.numpy.histogram_bin_edges.html#jax.numpy.histogram_bin_edges "jax.numpy.histogram_bin_edges"): Compute the bin edges for a histogram.

Examples

    >>> a = jnp.array([1, 2, 3, 10, 11, 15, 19, 25])
    >>> counts, bin_edges = jnp.histogram(a, bins=8)
    >>> print(counts)
    [3. 0. 0. 2. 1. 0. 1. 1.]
    >>> print(bin_edges)
    [ 1.  4.  7. 10. 13. 16. 19. 22. 25.]

Specifying the bin range:

    >>> counts, bin_edges = jnp.histogram(a, range=(0, 25), bins=5)
    >>> print(counts)
    [3. 0. 2. 2. 1.]
    >>> print(bin_edges)
    [ 0.  5. 10. 15. 20. 25.]

Specifying the bin edges explicitly:

    >>> bin_edges = jnp.array([0, 10, 20, 30])
    >>> counts, _ = jnp.histogram(a, bins=bin_edges)
    >>> print(counts)
    [3. 4. 1.]

Using `density=True` returns a normalized histogram:

    >>> density, bin_edges = jnp.histogram(a, density=True)
    >>> dx = jnp.diff(bin_edges)
    >>> normed_sum = jnp.sum(density * dx)
    >>> jnp.allclose(normed_sum, 1.0)
    Array(True, dtype=bool)

[](jax.numpy.heaviside.html "previous page")

previous

jax.numpy.heaviside

[](jax.numpy.histogram_bin_edges.html "next page")

next

jax.numpy.histogram_bin_edges

Contents

- [`histogram()`](#jax.numpy.histogram)

By The JAX authors

© Copyright 2024, The JAX Authors.\
