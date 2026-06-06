- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.geomspace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.geomspace.rst "Download source file")
-  .pdf

# jax.numpy.geomspace

## Contents

- [`geomspace()`](#jax.numpy.geomspace)

# jax.numpy.geomspace[\#](#jax-numpy-geomspace "Link to this heading")

jax.numpy.geomspace(*start*, *stop*, *num=50*, *endpoint=True*, *dtype=None*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_creation.py#L716-L768)[\#](#jax.numpy.geomspace "Link to this definition")  
Generate geometrically-spaced values.

JAX implementation of [`numpy.geomspace()`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html#numpy.geomspace "(in NumPy v2.4)").

Parameters:  
- **start** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array. Specifies the starting values.

- **stop** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array. Specifies the stop values.

- **num** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional, default=50. Number of values to generate.

- **endpoint** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, optional, default=True. If True, then include the `stop` value in the result. If False, then exclude the `stop` value.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – optional. Specifies the dtype of the output.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional, default=0. Axis along which to generate the geomspace.

Returns:  
An array containing the geometrically-spaced values.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.arange()`](jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange"): Generate `N` evenly-spaced values given a starting point and a step value.

- [`jax.numpy.linspace()`](jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace"): Generate evenly-spaced values.

- [`jax.numpy.logspace()`](jax.numpy.logspace.html#jax.numpy.logspace "jax.numpy.logspace"): Generate logarithmically-spaced values.

Examples

List 5 geometrically-spaced values between 1 and 16:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.geomspace(1, 16, 5)
    Array([ 1.,  2.,  4.,  8., 16.], dtype=float32)

List 4 geomtrically-spaced values between 1 and 16, with `endpoint=False`:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.geomspace(1, 16, 4, endpoint=False)
    Array([1., 2., 4., 8.], dtype=float32)

Multi-dimensional geomspace:

    >>> start = jnp.array([1, 1000])
    >>> stop = jnp.array([27, 1])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.geomspace(start, stop, 4)
    Array([[   1., 1000.],
           [   3.,  100.],
           [   9.,   10.],
           [  27.,    1.]], dtype=float32)

[](jax.numpy.generic.html "previous page")

previous

jax.numpy.generic

[](jax.numpy.get_printoptions.html "next page")

next

jax.numpy.get_printoptions

Contents

- [`geomspace()`](#jax.numpy.geomspace)

By The JAX authors

© Copyright 2024, The JAX Authors.\
