- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.psend

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.psend.rst "Download source file")
-  .pdf

# jax.lax.psend

## Contents

- [`psend()`](#jax.lax.psend)

# jax.lax.psend[\#](#jax-lax-psend "Link to this heading")

jax.lax.psend(*x*, *axis_name*, *perm*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L393-L426)[\#](#jax.lax.psend "Link to this definition")  
Perform a collective send according to the permutation `perm`.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

This function is an analog of the Send HLO.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **perm** – list of pairs of ints, representing `(source_index,`` ``destination_index)` pairs that encode how the mapped axis named `axis_name` should be shuffled. The integer values are treated as indices into the mapped axis `axis_name`. Any two pairs should not have the same source index or the same destination index. For each index of the axis `axis_name` that does not correspond to a destination index in `perm`, the corresponding values in the result are filled with zeros of the appropriate type. The semantics here are platform-specific, and for GPU they correspond to NCCL send.

Returns:  
A compiler token that can be used by precv and lax.optimzation_barrier to enforce ordering of collective ops.

[](jax.lax.axis_size.html "previous page")

previous

jax.lax.axis_size

[](jax.lax.precv.html "next page")

next

jax.lax.precv

Contents

- [`psend()`](#jax.lax.psend)

By The JAX authors

© Copyright 2024, The JAX Authors.\
