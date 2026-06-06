- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.pswapaxes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.pswapaxes.rst "Download source file")
-  .pdf

# jax.lax.pswapaxes

## Contents

- [`pswapaxes()`](#jax.lax.pswapaxes)

# jax.lax.pswapaxes[\#](#jax-lax-pswapaxes "Link to this heading")

jax.lax.pswapaxes(*x*, *axis_name*, *axis*, *\**, *axis_index_groups=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L490-L520)[\#](#jax.lax.pswapaxes "Link to this definition")  
Swap the pmapped axis `axis_name` with the unmapped axis `axis`.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

The group size of the mapped axis size must be equal to the size of the unmapped axis; that is, we must have `lax.psum(1,`` ``axis_name,`` ``axis_index_groups=axis_index_groups)`` ``==`` ``x.shape[axis]`. By default, when `axis_index_groups=None`, this encompasses all the devices.

This function is a special case of `all_to_all` where the pmapped axis of the input is placed at the position `axis` in the output. That is, it is equivalent to `all_to_all(x,`` ``axis_name,`` ``axis,`` ``axis)`.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **axis** – int indicating the unmapped axis of `x` to map with the name `axis_name`.

- **axis_index_groups** – optional list of lists containing axis indices (e.g. for an axis of size 4, \[\[0, 1\], \[2, 3\]\] would run pswapaxes over the first two and last two replicas). Groups must cover all axis indices exactly once, and all groups must be the same size.

Returns:  
Array(s) with the same shape as `x`.

[](jax.lax.pshuffle.html "previous page")

previous

jax.lax.pshuffle

[](jax.lax.axis_index.html "next page")

next

jax.lax.axis_index

Contents

- [`pswapaxes()`](#jax.lax.pswapaxes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
