- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.vsplit

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.vsplit.rst "Download source file")
-  .pdf

# jax.numpy.vsplit

## Contents

- [`vsplit()`](#jax.numpy.vsplit)

# jax.numpy.vsplit[\#](#jax-numpy-vsplit "Link to this heading")

jax.numpy.vsplit(*ary*, *indices_or_sections*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3218-L3251)[\#](#jax.numpy.vsplit "Link to this definition")  
Split an array into sub-arrays vertically.

JAX implementation of [`numpy.vsplit()`](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html#numpy.vsplit "(in NumPy v2.4)").

Refer to the documentation of [`jax.numpy.split()`](jax.numpy.split.html#jax.numpy.split "jax.numpy.split") for details; `vsplit` is equivalent to `split` with `axis=0`.

Examples

1D array:

    >>> x = jnp.array([1, 2, 3, 4, 5, 6])
    >>> x1, x2 = jnp.vsplit(x, 2)
    >>> print(x1, x2)
    [1 2 3] [4 5 6]

2D array:

    >>> x = jnp.array([[1, 2, 3, 4],
    ...                [5, 6, 7, 8]])
    >>> x1, x2 = jnp.vsplit(x, 2)
    >>> print(x1, x2)
    [[1 2 3 4]] [[5 6 7 8]]

See also

- [`jax.numpy.split()`](jax.numpy.split.html#jax.numpy.split "jax.numpy.split"): split an array along any axis.

- [`jax.numpy.hsplit()`](jax.numpy.hsplit.html#jax.numpy.hsplit "jax.numpy.hsplit"): split horizontally, i.e. along axis=1

- [`jax.numpy.dsplit()`](jax.numpy.dsplit.html#jax.numpy.dsplit "jax.numpy.dsplit"): split depth-wise, i.e. along axis=2

- [`jax.numpy.array_split()`](jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split"): like `split`, but allows `indices_or_sections` to be an integer that does not evenly divide the size of the array.

Parameters:  
- **ary** (*ArrayLike*)

- **indices_or_sections** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *ArrayLike*)

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.numpy.vectorize.html "previous page")

previous

jax.numpy.vectorize

[](jax.numpy.vstack.html "next page")

next

jax.numpy.vstack

Contents

- [`vsplit()`](#jax.numpy.vsplit)

By The JAX authors

© Copyright 2024, The JAX Authors.\
