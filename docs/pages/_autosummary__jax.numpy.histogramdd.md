- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.histogramdd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.histogramdd.rst "Download source file")
-  .pdf

# jax.numpy.histogramdd

## Contents

- [`histogramdd()`](#jax.numpy.histogramdd)

# jax.numpy.histogramdd[\#](#jax-numpy-histogramdd "Link to this heading")

jax.numpy.histogramdd(*sample*, *bins=10*, *range=None*, *weights=None*, *density=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L979-L1086)[\#](#jax.numpy.histogramdd "Link to this definition")  
Compute an N-dimensional histogram.

JAX implementation of [`numpy.histogramdd()`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html#numpy.histogramdd "(in NumPy v2.4)").

Parameters:  
- **sample** (*ArrayLike*) – input array of shape `(N,`` ``D)` representing `N` points in `D` dimensions.

- **bins** (*ArrayLike* *\|* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[ArrayLike\]*) – Specify the number of bins in each dimension of the histogram. (default: 10). May also be a length-D sequence of integers or arrays of bin edges.

- **range** (*Sequence\[None* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]\]* *\|* *None*) – Length-D sequence of pairs specifying the range for each dimension. If not specified, the range is inferred from the data.

- **weights** (*ArrayLike* *\|* *None*) – An optional shape `(N,)` array specifying the weights of the data points. Should be the same shape as `sample`. If not specified, each data point is weighted equally.

- **density** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – If True, return the normalized histogram in units of counts per unit volume. If False (default) return the (weighted) counts per bin.

Returns:  
A tuple of arrays `(histogram,`` ``bin_edges)`, where `histogram` contains the aggregated data, and `bin_edges` specifies the boundaries of the bins.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]\]

See also

- [`jax.numpy.histogram()`](jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram"): Compute the histogram of a 1D array.

- [`jax.numpy.histogram2d()`](jax.numpy.histogram2d.html#jax.numpy.histogram2d "jax.numpy.histogram2d"): Compute the histogram of a 2D array.

- [`jax.numpy.histogram_bin_edges()`](jax.numpy.histogram_bin_edges.html#jax.numpy.histogram_bin_edges "jax.numpy.histogram_bin_edges"): Compute the bin edges for a histogram.

Examples

A histogram over 100 points in three dimensions

    >>> key = jax.random.key(42)
    >>> a = jax.random.normal(key, (100, 3))
    >>> counts, bin_edges = jnp.histogramdd(a, bins=6,
    ...                                     range=[(-3, 3), (-3, 3), (-3, 3)])
    >>> counts.shape
    (6, 6, 6)
    >>> bin_edges  
    [Array([-3., -2., -1.,  0.,  1.,  2.,  3.], dtype=float32),
     Array([-3., -2., -1.,  0.,  1.,  2.,  3.], dtype=float32),
     Array([-3., -2., -1.,  0.,  1.,  2.,  3.], dtype=float32)]

Using `density=True` returns a normalized histogram:

    >>> density, bin_edges = jnp.histogramdd(a, density=True)
    >>> bin_widths = map(jnp.diff, bin_edges)
    >>> dx, dy, dz = jnp.meshgrid(*bin_widths, indexing='ij')
    >>> normed = jnp.sum(density * dx * dy * dz)
    >>> jnp.allclose(normed, 1.0)
    Array(True, dtype=bool)

[](jax.numpy.histogram2d.html "previous page")

previous

jax.numpy.histogram2d

[](jax.numpy.hsplit.html "next page")

next

jax.numpy.hsplit

Contents

- [`histogramdd()`](#jax.numpy.histogramdd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
