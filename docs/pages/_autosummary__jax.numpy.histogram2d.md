- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.histogram2d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.histogram2d.rst "Download source file")
-  .pdf

# jax.numpy.histogram2d

## Contents

- [`histogram2d()`](#jax.numpy.histogram2d)

# jax.numpy.histogram2d[\#](#jax-numpy-histogram2d "Link to this heading")

jax.numpy.histogram2d(*x*, *y*, *bins=10*, *range=None*, *weights=None*, *density=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L889-L977)[\#](#jax.numpy.histogram2d "Link to this definition")  
Compute a 2-dimensional histogram.

JAX implementation of [`numpy.histogram2d()`](https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html#numpy.histogram2d "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – one-dimensional array of x-values for points to be binned.

- **y** (*ArrayLike*) – one-dimensional array of y-values for points to be binned.

- **bins** (*ArrayLike* *\|* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[ArrayLike\]*) – Specify the number of bins in the histogram (default: 10). `bins` may also be an array specifying the locations of the bin edges, or a pair of integers or pair of arrays specifying the number of bins in each dimension.

- **range** (*Sequence\[None* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]\]* *\|* *None*) – Pair of arrays or lists of the form `[[xmin,`` ``xmax],`` ``[ymin,`` ``ymax]]` specifying the range of the data in each dimension. If not specified, the range is inferred from the data.

- **weights** (*ArrayLike* *\|* *None*) – An optional array specifying the weights of the data points. Should be the same shape as `x` and `y`. If not specified, each data point is weighted equally.

- **density** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – If True, return the normalized histogram in units of counts per unit area. If False (default) return the (weighted) counts per bin.

Returns:  
A tuple of arrays `(histogram,`` ``x_edges,`` ``y_edges)`, where `histogram` contains the aggregated data, and `x_edges` and `y_edges` specify the boundaries of the bins.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.histogram()`](jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram"): Compute the histogram of a 1D array.

- [`jax.numpy.histogramdd()`](jax.numpy.histogramdd.html#jax.numpy.histogramdd "jax.numpy.histogramdd"): Compute the histogram of an N-dimensional array.

- [`jax.numpy.histogram_bin_edges()`](jax.numpy.histogram_bin_edges.html#jax.numpy.histogram_bin_edges "jax.numpy.histogram_bin_edges"): Compute the bin edges for a histogram.

Examples

    >>> x = jnp.array([1, 2, 3, 10, 11, 15, 19, 25])
    >>> y = jnp.array([2, 5, 6, 8, 13, 16, 17, 18])
    >>> counts, x_edges, y_edges = jnp.histogram2d(x, y, bins=8)
    >>> counts.shape
    (8, 8)
    >>> x_edges
    Array([ 1.,  4.,  7., 10., 13., 16., 19., 22., 25.], dtype=float32)
    >>> y_edges
    Array([ 2.,  4.,  6.,  8., 10., 12., 14., 16., 18.], dtype=float32)

Specifying the bin range:

    >>> counts, x_edges, y_edges = jnp.histogram2d(x, y, range=[(0, 25), (0, 25)], bins=5)
    >>> counts.shape
    (5, 5)
    >>> x_edges
    Array([ 0.,  5., 10., 15., 20., 25.], dtype=float32)
    >>> y_edges
    Array([ 0.,  5., 10., 15., 20., 25.], dtype=float32)

Specifying the bin edges explicitly:

    >>> x_edges = jnp.array([0, 10, 20, 30])
    >>> y_edges = jnp.array([0, 10, 20, 30])
    >>> counts, _, _ = jnp.histogram2d(x, y, bins=[x_edges, y_edges])
    >>> counts
    Array([[3, 0, 0],
           [1, 3, 0],
           [0, 1, 0]], dtype=int32)

Using `density=True` returns a normalized histogram:

    >>> density, x_edges, y_edges = jnp.histogram2d(x, y, density=True)
    >>> dx = jnp.diff(x_edges)
    >>> dy = jnp.diff(y_edges)
    >>> normed_sum = jnp.sum(density * dx[:, None] * dy[None, :])
    >>> jnp.allclose(normed_sum, 1.0)
    Array(True, dtype=bool)

[](jax.numpy.histogram_bin_edges.html "previous page")

previous

jax.numpy.histogram_bin_edges

[](jax.numpy.histogramdd.html "next page")

next

jax.numpy.histogramdd

Contents

- [`histogram2d()`](#jax.numpy.histogram2d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
