- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.ndimage.map_coordinates

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.ndimage.map_coordinates.rst "Download source file")
-  .pdf

# jax.scipy.ndimage.map_coordinates

## Contents

- [`map_coordinates()`](#jax.scipy.ndimage.map_coordinates)

# jax.scipy.ndimage.map_coordinates[\#](#jax-scipy-ndimage-map-coordinates "Link to this heading")

jax.scipy.ndimage.map_coordinates(*input*, *coordinates*, *order*, *mode='constant'*, *cval=0.0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/ndimage.py#L128-L181)[\#](#jax.scipy.ndimage.map_coordinates "Link to this definition")  
Map the input array to new coordinates using interpolation.

JAX implementation of [`scipy.ndimage.map_coordinates()`](https://scipy.github.io/devdocs/reference/generated/scipy.ndimage.map_coordinates.html#scipy.ndimage.map_coordinates "(in SciPy v1.19.0.dev)")

Given an input array and a set of coordinates, this function returns the interpolated values of the input array at those coordinates.

Parameters:  
- **input** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – N-dimensional input array from which values are interpolated.

- **coordinates** ([*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")*\]*) – length-N sequence of arrays specifying the coordinates at which to evaluate the interpolated values

- **order** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) –

  The order of interpolation. JAX supports the following:

  - 0: Nearest-neighbor

  - 1: Linear

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Points outside the boundaries of the input are filled according to the given mode. JAX supports one of `('constant',`` ``'nearest',`` ``'mirror',`` ``'wrap',`` ``'reflect')`. Note the `'wrap'` mode in JAX behaves as `'grid-wrap'` mode in SciPy, and `'constant'` mode in JAX behaves as `'grid-constant'` mode in SciPy. This discrepancy was caused by a former bug in those modes in SciPy ([scipy/scipy#2640](https://github.com/scipy/scipy/issues/2640)), which was first fixed in JAX by changing the behavior of the existing modes, and later on fixed in SciPy, by adding modes with new names, rather than fixing the existing ones, for backwards compatibility reasons. Default is ‘constant’.

- **cval** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – Value used for points outside the boundaries of the input if `mode='constant'` Default is 0.0.

Returns:  
The interpolated values at the specified coordinates.

Examples

    >>> input = jnp.arange(12.0).reshape(3, 4)
    >>> input
    Array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]], dtype=float32)
    >>> coordinates = [jnp.array([0.5, 1.5]),
    ...                jnp.array([1.5, 2.5])]
    >>> jax.scipy.ndimage.map_coordinates(input, coordinates, order=1)
    Array([3.5, 8.5], dtype=float32)

Note

Interpolation near boundaries differs from the scipy function, because JAX fixed an outstanding bug; see [jax-ml/jax#11097](https://github.com/jax-ml/jax/issues/11097). This function interprets the `mode` argument as documented by SciPy, but not as implemented by SciPy.

[](jax.scipy.linalg.toeplitz.html "previous page")

previous

jax.scipy.linalg.toeplitz

[](jax.scipy.optimize.minimize.html "next page")

next

jax.scipy.optimize.minimize

Contents

- [`map_coordinates()`](#jax.scipy.ndimage.map_coordinates)

By The JAX authors

© Copyright 2024, The JAX Authors.\
