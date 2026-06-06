- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.histogram_bin_edges

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.histogram_bin_edges.rst "Download source file")
-  .pdf

# jax.numpy.histogram_bin_edges

## Contents

- [`histogram_bin_edges()`](#jax.numpy.histogram_bin_edges)

# jax.numpy.histogram_bin_edges[\#](#jax-numpy-histogram-bin-edges "Link to this heading")

jax.numpy.histogram_bin_edges(*a*, *bins=10*, *range=None*, *weights=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L755-L805)[\#](#jax.numpy.histogram_bin_edges "Link to this definition")  
Compute the bin edges for a histogram.

JAX implementation of [`numpy.histogram_bin_edges()`](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array of values to be binned

- **bins** (*ArrayLike*) – Specify the number of bins in the histogram (default: 10).

- **range** (*None* *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – tuple of scalars. Specifies the range of the data. If not specified, the range is inferred from the data.

- **weights** (*ArrayLike* *\|* *None*) – unused by JAX.

Returns:  
An array of bin edges for the histogram.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.histogram()`](jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram"): compute a 1D histogram.

- [`jax.numpy.histogram2d()`](jax.numpy.histogram2d.html#jax.numpy.histogram2d "jax.numpy.histogram2d"): compute a 2D histogram.

- [`jax.numpy.histogramdd()`](jax.numpy.histogramdd.html#jax.numpy.histogramdd "jax.numpy.histogramdd"): compute an N-dimensional histogram.

Examples

    >>> a = jnp.array([2, 5, 3, 6, 4, 1])
    >>> jnp.histogram_bin_edges(a, bins=5)
    Array([1., 2., 3., 4., 5., 6.], dtype=float32)
    >>> jnp.histogram_bin_edges(a, bins=5, range=(-10, 10))  
    Array([-10.,  -6.,  -2.,   2.,   6.,  10.], dtype=float32)

[](jax.numpy.histogram.html "previous page")

previous

jax.numpy.histogram

[](jax.numpy.histogram2d.html "next page")

next

jax.numpy.histogram2d

Contents

- [`histogram_bin_edges()`](#jax.numpy.histogram_bin_edges)

By The JAX authors

© Copyright 2024, The JAX Authors.\
