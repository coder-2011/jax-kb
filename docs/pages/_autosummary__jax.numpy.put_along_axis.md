- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.put_along_axis

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.put_along_axis.rst "Download source file")
-  .pdf

# jax.numpy.put_along_axis

## Contents

- [`put_along_axis()`](#jax.numpy.put_along_axis)

# jax.numpy.put_along_axis[\#](#jax-numpy-put-along-axis "Link to this heading")

jax.numpy.put_along_axis(*arr*, *indices*, *values*, *axis*, *inplace=True*, *\**, *mode=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/indexing.py#L957-L1045)[\#](#jax.numpy.put_along_axis "Link to this definition")  
Put values into the destination array by matching 1d index and data slices.

JAX implementation of [`numpy.put_along_axis()`](https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html#numpy.put_along_axis "(in NumPy v2.4)").

The semantics of [`numpy.put_along_axis()`](https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html#numpy.put_along_axis "(in NumPy v2.4)") are to modify arrays in-place, which is not possible for JAX’s immutable arrays. The JAX version returns a modified copy of the input, and adds the `inplace` parameter which must be set to False\` by the user as a reminder of this API difference.

Parameters:  
- **arr** (*ArrayLike*) – array into which values will be put.

- **indices** (*ArrayLike*) – array of indices at which to put values.

- **values** (*ArrayLike*) – array of values to put into the array.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – the axis along which to put values. If not specified, the array will be flattened before indexing is applied.

- **inplace** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – must be set to False to indicate that the input is not modified in-place, but rather a modified copy is returned.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – Out-of-bounds indexing mode. For more discussion of `mode` options, see [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at").

Returns:  
A copy of `a` with specified entries updated.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.put()`](jax.numpy.put.html#jax.numpy.put "jax.numpy.put"): put elements into an array at given indices.

- [`jax.numpy.place()`](jax.numpy.place.html#jax.numpy.place "jax.numpy.place"): place elements into an array via boolean mask.

- [`jax.numpy.ndarray.at()`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"): array updates using NumPy-style indexing.

- [`jax.numpy.take()`](jax.numpy.take.html#jax.numpy.take "jax.numpy.take"): extract values from an array at given indices.

- [`jax.numpy.take_along_axis()`](jax.numpy.take_along_axis.html#jax.numpy.take_along_axis "jax.numpy.take_along_axis"): extract values from an array along an axis.

Examples

    >>> from jax import numpy as jnp
    >>> a = jnp.array([[10, 30, 20], [60, 40, 50]])
    >>> i = jnp.argmax(a, axis=1, keepdims=True)
    >>> print(i)
    [[1]
     [0]]
    >>> b = jnp.put_along_axis(a, i, 99, axis=1, inplace=False)
    >>> print(b)
    [[10 99 20]
     [99 40 50]]

[](jax.numpy.put.html "previous page")

previous

jax.numpy.put

[](jax.numpy.quantile.html "next page")

next

jax.numpy.quantile

Contents

- [`put_along_axis()`](#jax.numpy.put_along_axis)

By The JAX authors

© Copyright 2024, The JAX Authors.\
