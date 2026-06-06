- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.pmin

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.pmin.rst "Download source file")
-  .pdf

# jax.lax.pmin

## Contents

- [`pmin()`](#jax.lax.pmin)

# jax.lax.pmin[\#](#jax-lax-pmin "Link to this heading")

jax.lax.pmin(*x*, *axis_name*, *\**, *axis_index_groups=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L260-L289)[\#](#jax.lax.pmin "Link to this definition")  
Compute an all-reduce min on `x` over the pmapped axis `axis_name`.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **axis_index_groups** – optional list of lists containing axis indices (e.g. for an axis of size 4, \[\[0, 1\], \[2, 3\]\] would perform pmins over the first two and last two replicas). Groups must cover all axis indices exactly once, and on TPUs all groups must be the same size.

Returns:  
Array(s) with the same shape as `x` representing the result of an all-reduce min along the axis `axis_name`.

[](jax.lax.pmax.html "previous page")

previous

jax.lax.pmax

[](jax.lax.pmean.html "next page")

next

jax.lax.pmean

Contents

- [`pmin()`](#jax.lax.pmin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
