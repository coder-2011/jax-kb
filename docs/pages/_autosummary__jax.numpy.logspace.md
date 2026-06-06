- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.logspace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.logspace.rst "Download source file")
-  .pdf

# jax.numpy.logspace

## Contents

- [`logspace()`](#jax.numpy.logspace)

# jax.numpy.logspace[\#](#jax-numpy-logspace "Link to this heading")

jax.numpy.logspace(*start*, *stop*, *num=50*, *endpoint=True*, *base=10.0*, *dtype=None*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_creation.py#L630-L697)[\#](#jax.numpy.logspace "Link to this definition")  
Generate logarithmically-spaced values.

JAX implementation of [`numpy.logspace()`](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html#numpy.logspace "(in NumPy v2.4)").

Parameters:  
- **start** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array. Used to specify the start value. The start value is `base`` ``**`` ``start`.

- **stop** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array. Used to specify the stop value. The end value is `base`` ``**`` ``stop`.

- **num** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional, default=50. Number of values to generate.

- **endpoint** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, optional, default=True. If True, then include the `stop` value in the result. If False, then exclude the `stop` value.

- **base** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array, optional, default=10. Specifies the base of the logarithm.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*) – optional. Specifies the dtype of the output.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, optional, default=0. Axis along which to generate the logspace.

Returns:  
An array of logarithm.

Return type:  
[*Array*](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.arange()`](jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange"): Generate `N` evenly-spaced values given a starting point and a step value.

- [`jax.numpy.linspace()`](jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace"): Generate evenly-spaced values.

- [`jax.numpy.geomspace()`](jax.numpy.geomspace.html#jax.numpy.geomspace "jax.numpy.geomspace"): Generate geometrically-spaced values.

Examples

List 5 logarithmically spaced values between 1 (`10`` ``**`` ``0`) and 100 (`10`` ``**`` ``2`):

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(0, 2, 5)
    Array([  1.   ,   3.162,  10.   ,  31.623, 100.   ], dtype=float32)

List 5 logarithmically-spaced values between 1(`10`` ``**`` ``0`) and 100 (`10`` ``**`` ``2`), excluding endpoint:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(0, 2, 5, endpoint=False)
    Array([ 1.   ,  2.512,  6.31 , 15.849, 39.811], dtype=float32)

List 7 logarithmically-spaced values between 1 (`2`` ``**`` ``0`) and 4 (`2`` ``**`` ``2`) with base 2:

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(0, 2, 7, base=2)
    Array([1.   , 1.26 , 1.587, 2.   , 2.52 , 3.175, 4.   ], dtype=float32)

Multi-dimensional logspace:

    >>> start = jnp.array([0, 5])
    >>> stop = jnp.array([5, 0])
    >>> base = jnp.array([2, 3])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.logspace(start, stop, 5, base=base)
    Array([[  1.   , 243.   ],
           [  2.378,  61.547],
           [  5.657,  15.588],
           [ 13.454,   3.948],
           [ 32.   ,   1.   ]], dtype=float32)

[](jax.numpy.logical_xor.html "previous page")

previous

jax.numpy.logical_xor

[](jax.numpy.mask_indices.html "next page")

next

jax.numpy.mask_indices

Contents

- [`logspace()`](#jax.numpy.logspace)

By The JAX authors

© Copyright 2024, The JAX Authors.\
