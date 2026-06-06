- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.split

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.split.rst "Download source file")
-  .pdf

# jax.numpy.split

## Contents

- [`split()`](#jax.numpy.split)

# jax.numpy.split[\#](#jax-numpy-split "Link to this heading")

jax.numpy.split(*ary*, *indices_or_sections*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3154-L3216)[\#](#jax.numpy.split "Link to this definition")  
Split an array into sub-arrays.

JAX implementation of [`numpy.split()`](https://numpy.org/doc/stable/reference/generated/numpy.split.html#numpy.split "(in NumPy v2.4)").

Parameters:  
- **ary** (*ArrayLike*) – N-dimensional array-like object to split

- **indices_or_sections** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *ArrayLike*) –

  either a single integer or a sequence of indices.

  - if `indices_or_sections` is an integer *N*, then *N* must evenly divide `ary.shape[axis]` and `ary` will be divided into *N* equally-sized chunks along `axis`.

  - if `indices_or_sections` is a sequence of integers, then these integers specify the boundary between unevenly-sized chunks along `axis`; see examples below.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to split; defaults to 0.

Returns:  
A list of arrays. If `indices_or_sections` is an integer *N*, then the list is of length *N*. If `indices_or_sections` is a sequence *seq*, then the list is is of length *len(seq) + 1*.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

Examples

Splitting a 1-dimensional array:

    >>> x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

Split into three equal sections:

    >>> chunks = jnp.split(x, 3)
    >>> print(*chunks)
    [1 2 3] [4 5 6] [7 8 9]

Split into sections by index:

    >>> chunks = jnp.split(x, [2, 7])  # [x[0:2], x[2:7], x[7:]]
    >>> print(*chunks)
    [1 2] [3 4 5 6 7] [8 9]

Splitting a two-dimensional array along axis 1:

    >>> x = jnp.array([[1, 2, 3, 4],
    ...                [5, 6, 7, 8]])
    >>> x1, x2 = jnp.split(x, 2, axis=1)
    >>> print(x1)
    [[1 2]
     [5 6]]
    >>> print(x2)
    [[3 4]
     [7 8]]

See also

- [`jax.numpy.array_split()`](jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split"): like `split`, but allows `indices_or_sections` to be an integer that does not evenly divide the size of the array.

- [`jax.numpy.vsplit()`](jax.numpy.vsplit.html#jax.numpy.vsplit "jax.numpy.vsplit"): split vertically, i.e. along axis=0

- [`jax.numpy.hsplit()`](jax.numpy.hsplit.html#jax.numpy.hsplit "jax.numpy.hsplit"): split horizontally, i.e. along axis=1

- [`jax.numpy.dsplit()`](jax.numpy.dsplit.html#jax.numpy.dsplit "jax.numpy.dsplit"): split depth-wise, i.e. along axis=2

[](jax.numpy.spacing.html "previous page")

previous

jax.numpy.spacing

[](jax.numpy.sqrt.html "next page")

next

jax.numpy.sqrt

Contents

- [`split()`](#jax.numpy.split)

By The JAX authors

© Copyright 2024, The JAX Authors.\
