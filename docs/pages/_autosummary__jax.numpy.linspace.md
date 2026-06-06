- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linspace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linspace.rst "Download source file")
-  .pdf

# jax.numpy.linspace

## Contents

- [`linspace()`](#jax.numpy.linspace)

# jax.numpy.linspace[\#](#jax-numpy-linspace "Link to this heading")

jax.numpy.linspace(*start*, *stop*, *num=50*, *endpoint=True*, *retstep=False*, *dtype=None*, *axis=0*, *\**, *device=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_creation.py#L504-L571)[\#](#jax.numpy.linspace "Link to this definition")  
Return evenly-spaced numbers within an interval.

JAX implementation of [`numpy.linspace()`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace "(in NumPy v2.4)").

Parameters:  
- **start** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array of starting values.

- **stop** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – scalar or array of stop values.

- **num** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – number of values to generate. Default: 50.

- **endpoint** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default) then include the `stop` value in the result. If False, then exclude the `stop` value.

- **retstep** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, then return a `(result,`` ``step)` tuple, where `step` is the interval between adjacent values in `result`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer axis along which to generate the linspace. Defaults to zero.

- **device** ([*Device*](jax.Device.html#jax.Device "jaxlib._jax.Device") *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jaxlib._jax.Sharding") *\|* *None*) – optional [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

- **dtype** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[*[*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")*\]* *\|* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") *\|* *SupportsDType* *\|* *None*)

Returns:  
- `values` is an array of evenly-spaced values from `start` to `stop`

- `step` is the interval between adjacent values.

Return type:  
An array `values`, or a tuple `(values,`` ``step)` if `retstep` is True, where

See also

- [`jax.numpy.arange()`](jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange"): Generate `N` evenly-spaced values given a starting point and a step

- [`jax.numpy.logspace()`](jax.numpy.logspace.html#jax.numpy.logspace "jax.numpy.logspace"): Generate logarithmically-spaced values.

- [`jax.numpy.geomspace()`](jax.numpy.geomspace.html#jax.numpy.geomspace "jax.numpy.geomspace"): Generate geometrically-spaced values.

Examples

List of 5 values between 0 and 10:

    >>> jnp.linspace(0, 10, 5)
    Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32)

List of 8 values between 0 and 10, excluding the endpoint:

    >>> jnp.linspace(0, 10, 8, endpoint=False)
    Array([0.  , 1.25, 2.5 , 3.75, 5.  , 6.25, 7.5 , 8.75], dtype=float32)

List of values and the step size between them

    >>> vals, step = jnp.linspace(0, 10, 9, retstep=True)
    >>> vals
    Array([ 0.  ,  1.25,  2.5 ,  3.75,  5.  ,  6.25,  7.5 ,  8.75, 10.  ],      dtype=float32)
    >>> step
    Array(1.25, dtype=float32)

Multi-dimensional linspace:

    >>> start = jnp.array([0, 5])
    >>> stop = jnp.array([5, 10])
    >>> jnp.linspace(start, stop, 5)
    Array([[ 0.  ,  5.  ],
           [ 1.25,  6.25],
           [ 2.5 ,  7.5 ],
           [ 3.75,  8.75],
           [ 5.  , 10.  ]], dtype=float32)

[](jax.numpy.lexsort.html "previous page")

previous

jax.numpy.lexsort

[](jax.numpy.load.html "next page")

next

jax.numpy.load

Contents

- [`linspace()`](#jax.numpy.linspace)

By The JAX authors

© Copyright 2024, The JAX Authors.\
