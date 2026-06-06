- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ufunc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ufunc.rst "Download source file")
-  .pdf

# jax.numpy.ufunc

## Contents

- [`ufunc`](#jax.numpy.ufunc)
  - [`ufunc.__init__()`](#jax.numpy.ufunc.__init__)

# jax.numpy.ufunc[\#](#jax-numpy-ufunc "Link to this heading")

*class* jax.numpy.ufunc(*func*, */*, *nin*, *nout*, *\**, *name=None*, *nargs=None*, *identity=None*, *call=None*, *reduce=None*, *accumulate=None*, *at=None*, *reduceat=None*)[\#](#jax.numpy.ufunc "Link to this definition")  
Universal functions which operation element-by-element on arrays.

JAX implementation of [`numpy.ufunc`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html#numpy.ufunc "(in NumPy v2.4)").

This is a class for JAX-backed implementations of NumPy’s ufunc APIs. Most users will never need to instantiate [`ufunc`](#jax.numpy.ufunc "jax.numpy.ufunc"), but rather will use the pre-defined ufuncs in [`jax.numpy`](../jax.numpy.html#module-jax.numpy "jax.numpy").

For constructing your own ufuncs, see [`jax.numpy.frompyfunc()`](jax.numpy.frompyfunc.html#jax.numpy.frompyfunc "jax.numpy.frompyfunc").

Examples

Universal functions are functions that apply element-wise to broadcasted arrays, but they also come with a number of extra attributes and methods.

As an example, consider the function [`jax.numpy.add`](jax.numpy.add.html#jax.numpy.add "jax.numpy.add"). The object acts as a function that applies addition to broadcasted arrays in an element-wise manner:

    >>> x = jnp.array([1, 2, 3, 4, 5])
    >>> jnp.add(x, 1)
    Array([2, 3, 4, 5, 6], dtype=int32)

Each [`ufunc`](#jax.numpy.ufunc "jax.numpy.ufunc") object includes a number of attributes that describe its behavior:

    >>> jnp.add.nin  # number of inputs
    2
    >>> jnp.add.nout  # number of outputs
    1
    >>> jnp.add.identity  # identity value, or None if no identity exists
    0

Binary ufuncs like [`jax.numpy.add`](jax.numpy.add.html#jax.numpy.add "jax.numpy.add") include number of methods to apply the function to arrays in different manners.

The `outer()` method applies the function to the pair-wise outer-product of the input array values:

    >>> jnp.add.outer(x, x)
    Array([[ 2,  3,  4,  5,  6],
           [ 3,  4,  5,  6,  7],
           [ 4,  5,  6,  7,  8],
           [ 5,  6,  7,  8,  9],
           [ 6,  7,  8,  9, 10]], dtype=int32)

The `ufunc.reduce()` method performs a reduction over the array. For example, `jnp.add.reduce()` is equivalent to `jnp.sum`:

    >>> jnp.add.reduce(x)
    Array(15, dtype=int32)

The `ufunc.accumulate()` method performs a cumulative reduction over the array. For example, `jnp.add.accumulate()` is equivalent to [`jax.numpy.cumulative_sum()`](jax.numpy.cumulative_sum.html#jax.numpy.cumulative_sum "jax.numpy.cumulative_sum"):

    >>> jnp.add.accumulate(x)
    Array([ 1,  3,  6, 10, 15], dtype=int32)

The `ufunc.at()` method applies the function at particular indices in the array; for `jnp.add` the computation is similar to [`jax.lax.scatter_add()`](jax.lax.scatter_add.html#jax.lax.scatter_add "jax.lax.scatter_add"):

    >>> jnp.add.at(x, 0, 100, inplace=False)
    Array([101,   2,   3,   4,   5], dtype=int32)

And the `ufunc.reduceat()` method performs a number of `reduce` operations between specified indices of an array; for `jnp.add` the operation is similar to [`jax.ops.segment_sum()`](jax.ops.segment_sum.html#jax.ops.segment_sum "jax.ops.segment_sum"):

    >>> jnp.add.reduceat(x, jnp.array([0, 2]))
    Array([ 3, 12], dtype=int32)

In this case, the first element is `x[0:2].sum()`, and the second element is `x[2:].sum()`.

Parameters:  
- **func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*)

- **nin** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **nout** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*)

- **nargs** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **identity** (*Any*)

- **call** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **reduce** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **accumulate** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **at** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **reduceat** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

\_\_init\_\_(*func*, */*, *nin*, *nout*, *\**, *name=None*, *nargs=None*, *identity=None*, *call=None*, *reduce=None*, *accumulate=None*, *at=None*, *reduceat=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufunc_api.py#L124-L153)[\#](#jax.numpy.ufunc.__init__ "Link to this definition")  
Parameters:  
- **func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*)

- **nin** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **nout** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*)

- **nargs** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **identity** (*Any*)

- **call** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **reduce** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **accumulate** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **at** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

- **reduceat** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]* *\|* *None*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.numpy.ufunc.__init__ "jax.numpy.ufunc.__init__")(func, /, nin, nout, \*\[, name, ...\]) |  |
| `accumulate`(a\[, axis, dtype, out\]) | Accumulate operation derived from binary ufunc. |
| `at`(a, indices\[, b, inplace\]) | Update elements of an array via the specified unary or binary ufunc. |
| `outer`(A, B, /) | Apply the function to all pairs of values in `A` and `B`. |
| `reduce`(a\[, axis, dtype, out, keepdims, ...\]) | Reduction operation derived from a binary function. |
| `reduceat`(a, indices\[, axis, dtype, out\]) | Reduce an array between specified indices via a binary ufunc. |

Attributes

|            |     |
|------------|-----|
| `identity` |     |
| `nargs`    |     |
| `nin`      |     |
| `nout`     |     |

[](jax.numpy.trunc.html "previous page")

previous

jax.numpy.trunc

[](jax.numpy.uint.html "next page")

next

jax.numpy.uint

Contents

- [`ufunc`](#jax.numpy.ufunc)
  - [`ufunc.__init__()`](#jax.numpy.ufunc.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
