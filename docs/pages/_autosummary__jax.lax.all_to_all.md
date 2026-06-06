- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.all_to_all

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.all_to_all.rst "Download source file")
-  .pdf

# jax.lax.all_to_all

## Contents

- [`all_to_all()`](#jax.lax.all_to_all)

# jax.lax.all_to_all[\#](#jax-lax-all-to-all "Link to this heading")

jax.lax.all_to_all(*x*, *axis_name*, *split_axis*, *concat_axis*, *\**, *axis_index_groups=None*, *tiled=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L521-L567)[\#](#jax.lax.all_to_all "Link to this definition")  
Materialize the mapped axis and map a different axis.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

In the output, the input mapped axis `axis_name` is materialized at the logical axis position `concat_axis`, and the input unmapped axis at position `split_axis` is mapped with the name `axis_name`.

The group size of the mapped axis size must be equal to the size of the unmapped axis; that is, we must have `lax.psum(1,`` ``axis_name,`` ``axis_index_groups=axis_index_groups)`` ``==`` ``x.shape[axis]`. By default, when `axis_index_groups=None`, this encompasses all the devices.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **split_axis** – int indicating the unmapped axis of `x` to map with the name `axis_name`.

- **concat_axis** – int indicating the position in the output to materialize the mapped axis of the input with the name `axis_name`.

- **axis_index_groups** – optional list of lists containing axis indices (e.g. for an axis of size 4, \[\[0, 1\], \[2, 3\]\] would run all_to_all over the first two and last two replicas). Groups must cover all axis indices exactly once, and all groups must be the same size.

- **tiled** – when True, all_to_all will divide split_axis into chunks and concatenate them along concat_axis. In particular, no dimensions are added or removed. False by default.

Returns:  
When tiled is False, array(s) with shape given by the expression:

    np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

where `axis_size` is the size of the mapped axis named `axis_name` in the input `x`.

Otherwise array with shape similar to the input shape, except with split_axis divided by axis size and concat_axis multiplied by axis size.

[](jax.lax.all_gather.html "previous page")

previous

jax.lax.all_gather

[](jax.lax.psum.html "next page")

next

jax.lax.psum

Contents

- [`all_to_all()`](#jax.lax.all_to_all)

By The JAX authors

© Copyright 2024, The JAX Authors.\
