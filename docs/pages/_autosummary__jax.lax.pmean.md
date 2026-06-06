- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.pmean

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.pmean.rst "Download source file")
-  .pdf

# jax.lax.pmean

## Contents

- [`pmean()`](#jax.lax.pmean)

# jax.lax.pmean[\#](#jax-lax-pmean "Link to this heading")

jax.lax.pmean(*x*, *axis_name*, *\**, *axis_index_groups=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L196-L228)[\#](#jax.lax.pmean "Link to this definition")  
Compute an all-reduce mean on `x` over the pmapped axis `axis_name`.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **axis_index_groups** – optional list of lists containing axis indices (e.g. for an axis of size 4, \[\[0, 1\], \[2, 3\]\] would perform pmeans over the first two and last two replicas). Groups must cover all axis indices exactly once, and on TPUs all groups must be the same size.

Returns:  
Array(s) with the same shape as `x` representing the result of an all-reduce mean along the axis `axis_name`.

For example, with 4 XLA devices available:

    >>> x = np.arange(4)
    >>> y = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [1.5 1.5 1.5 1.5]
    >>> y = jax.pmap(lambda x: x / jax.lax.pmean(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [0.        0.6666667 1.3333334 2.       ]

[](jax.lax.pmin.html "previous page")

previous

jax.lax.pmin

[](jax.lax.ppermute.html "next page")

next

jax.lax.ppermute

Contents

- [`pmean()`](#jax.lax.pmean)

By The JAX authors

© Copyright 2024, The JAX Authors.\
