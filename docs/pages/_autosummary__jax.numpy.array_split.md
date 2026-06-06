- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.array_split

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.array_split.rst "Download source file")
-  .pdf

# jax.numpy.array_split

## Contents

- [`array_split()`](#jax.numpy.array_split)

# jax.numpy.array_split[\#](#jax-numpy-array-split "Link to this heading")

jax.numpy.array_split(*ary*, *indices_or_sections*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3335-L3359)[\#](#jax.numpy.array_split "Link to this definition")  
Split an array into sub-arrays.

JAX implementation of [`numpy.array_split()`](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split "(in NumPy v2.4)").

Refer to the documentation of [`jax.numpy.split()`](jax.numpy.split.html#jax.numpy.split "jax.numpy.split") for details; `array_split` is equivalent to `split`, but allows integer `indices_or_sections` which does not evenly divide the split axis.

Examples

    >>> x = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> chunks = jnp.array_split(x, 4)
    >>> print(*chunks)
    [1 2 3] [4 5] [6 7] [8 9]

See also

- [`jax.numpy.split()`](jax.numpy.split.html#jax.numpy.split "jax.numpy.split"): split an array along any axis.

- [`jax.numpy.vsplit()`](jax.numpy.vsplit.html#jax.numpy.vsplit "jax.numpy.vsplit"): split vertically, i.e. along axis=0

- [`jax.numpy.hsplit()`](jax.numpy.hsplit.html#jax.numpy.hsplit "jax.numpy.hsplit"): split horizontally, i.e. along axis=1

- [`jax.numpy.dsplit()`](jax.numpy.dsplit.html#jax.numpy.dsplit "jax.numpy.dsplit"): split depth-wise, i.e. along axis=2

Parameters:  
- **ary** (*ArrayLike*)

- **indices_or_sections** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *ArrayLike*)

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.numpy.array_repr.html "previous page")

previous

jax.numpy.array_repr

[](jax.numpy.array_str.html "next page")

next

jax.numpy.array_str

Contents

- [`array_split()`](#jax.numpy.array_split)

By The JAX authors

© Copyright 2024, The JAX Authors.\
