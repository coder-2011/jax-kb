- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.interpolate.RegularGridInterpolator

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.interpolate.RegularGridInterpolator.rst "Download source file")
-  .pdf

# jax.scipy.interpolate.RegularGridInterpolator

## Contents

- [`RegularGridInterpolator`](#jax.scipy.interpolate.RegularGridInterpolator)
  - [`RegularGridInterpolator.__init__()`](#jax.scipy.interpolate.RegularGridInterpolator.__init__)

# jax.scipy.interpolate.RegularGridInterpolator[\#](#jax-scipy-interpolate-regulargridinterpolator "Link to this heading")

*class* jax.scipy.interpolate.RegularGridInterpolator(*points*, *values*, *method='linear'*, *bounds_error=False*, *fill_value=nan*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/third_party/scipy/interpolate.py#L36-L171)[\#](#jax.scipy.interpolate.RegularGridInterpolator "Link to this definition")  
Interpolate points on a regular rectangular grid.

JAX implementation of `scipy.interpolate.RegularGridInterpolator()`.

Parameters:  
- **points** – length-N sequence of arrays specifying the grid coordinates.

- **values** – N-dimensional array specifying the grid values.

- **method** – interpolation method, either `"linear"` or `"nearest"`.

- **bounds_error** – not implemented by JAX

- **fill_value** – value returned for points outside the grid, defaults to NaN.

Returns:  
callable interpolation object.

Return type:  
interpolator

Examples

    >>> points = (jnp.array([1, 2, 3]), jnp.array([4, 5, 6]))
    >>> values = jnp.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    >>> interpolate = RegularGridInterpolator(points, values, method='linear')

    >>> query_points = jnp.array([[1.5, 4.5], [2.2, 5.8]])
    >>> interpolate(query_points)
    Array([30., 64.], dtype=float32)

Note

Unlike [`scipy.interpolate.RegularGridInterpolator`](https://scipy.github.io/devdocs/reference/generated/scipy.interpolate.RegularGridInterpolator.html#scipy.interpolate.RegularGridInterpolator "(in SciPy v1.19.0.dev)"), JAX requires each axis in `points` to be strictly increasing. SciPy accepts any monotonic axis (increasing or decreasing), but a decreasing axis is not supported here and will silently produce incorrect results. Reorder the grid (and the corresponding axis of `values`) so each axis is strictly increasing before constructing the interpolator.

\_\_init\_\_(*points*, *values*, *method='linear'*, *bounds_error=False*, *fill_value=nan*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/third_party/scipy/interpolate.py#L71-L103)[\#](#jax.scipy.interpolate.RegularGridInterpolator.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.scipy.interpolate.RegularGridInterpolator.__init__ "jax.scipy.interpolate.RegularGridInterpolator.__init__")(points, values\[, method, ...\]) |  |

[](jax.scipy.integrate.trapezoid.html "previous page")

previous

jax.scipy.integrate.trapezoid

[](jax.scipy.linalg.block_diag.html "next page")

next

jax.scipy.linalg.block_diag

Contents

- [`RegularGridInterpolator`](#jax.scipy.interpolate.RegularGridInterpolator)
  - [`RegularGridInterpolator.__init__()`](#jax.scipy.interpolate.RegularGridInterpolator.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
