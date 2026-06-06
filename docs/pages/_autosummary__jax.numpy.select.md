- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.select

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.select.rst "Download source file")
-  .pdf

# jax.numpy.select

## Contents

- [`select()`](#jax.numpy.select)

# jax.numpy.select[\#](#jax-numpy-select "Link to this heading")

jax.numpy.select(*condlist*, *choicelist*, *default=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L2802-L2874)[\#](#jax.numpy.select "Link to this definition")  
Select values based on a series of conditions.

JAX implementation of [`numpy.select()`](https://numpy.org/doc/stable/reference/generated/numpy.select.html#numpy.select "(in NumPy v2.4)"), implemented in terms of [`jax.lax.select_n()`](jax.lax.select_n.html#jax.lax.select_n "jax.lax.select_n")

Parameters:  
- **condlist** (*Sequence\[ArrayLike\]*) – sequence of array-like conditions. All entries must be mutually broadcast-compatible.

- **choicelist** (*Sequence\[ArrayLike\]*) – sequence of array-like values to choose. Must have the same length as `condlist`, and all entries must be broadcast-compatible with entries of `condlist`.

- **default** (*ArrayLike*) – value to return when every condition is False (default: 0).

Returns:  
Array of selected values from `choicelist` corresponding to the first `True` entry in `condlist` at each location.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.where()`](jax.numpy.where.html#jax.numpy.where "jax.numpy.where"): select between two values based on a single condition.

- [`jax.lax.select_n()`](jax.lax.select_n.html#jax.lax.select_n "jax.lax.select_n"): select between *N* values based on an index.

Examples

    >>> condlist = [
    ...    jnp.array([False, True, False, False]),
    ...    jnp.array([True, False, False, False]),
    ...    jnp.array([False, True, True, False]),
    ... ]
    >>> choicelist = [
    ...    jnp.array([1, 2, 3, 4]),
    ...    jnp.array([10, 20, 30, 40]),
    ...    jnp.array([100, 200, 300, 400]),
    ... ]
    >>> jnp.select(condlist, choicelist, default=0)
    Array([ 10,   2, 300,   0], dtype=int32)

This is logically equivalent to the following nested `where` statement:

    >>> default = 0
    >>> jnp.where(condlist[0],
    ...   choicelist[0],
    ...   jnp.where(condlist[1],
    ...     choicelist[1],
    ...     jnp.where(condlist[2],
    ...       choicelist[2],
    ...       default)))
    Array([ 10,   2, 300,   0], dtype=int32)

However, for efficiency it is implemented in terms of [`jax.lax.select_n()`](jax.lax.select_n.html#jax.lax.select_n "jax.lax.select_n").

[](jax.numpy.searchsorted.html "previous page")

previous

jax.numpy.searchsorted

[](jax.numpy.set_printoptions.html "next page")

next

jax.numpy.set_printoptions

Contents

- [`select()`](#jax.numpy.select)

By The JAX authors

© Copyright 2024, The JAX Authors.\
