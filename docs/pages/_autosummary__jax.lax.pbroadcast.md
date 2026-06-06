- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.pbroadcast

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.pbroadcast.rst "Download source file")
-  .pdf

# jax.lax.pbroadcast

## Contents

- [`pbroadcast()`](#jax.lax.pbroadcast)

# jax.lax.pbroadcast[\#](#jax-lax-pbroadcast "Link to this heading")

jax.lax.pbroadcast(*x*, *axis_name*, *source*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L323-L349)[\#](#jax.lax.pbroadcast "Link to this definition")  
Perform a collective broadcast and replicate from `source`.

This is equivalent to:

    def pbroadcast(x, axis_name, source):
      masked = jnp.where(axis_index(axis_name) == source, x, zeros_like(x))
      return psum(masked, axis_name)

but implemented in a hardware optimized way.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

This function is an analog of the CollectiveBroadcast HLO.

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **source** – int, representing which index into `axis_name` that should be copied.

Returns:  
Array(s) with `x` being copied from the `source` index slice of `axis_name`.

[](jax.lax.precv.html "previous page")

previous

jax.lax.precv

[](jax.lax.with_sharding_constraint.html "next page")

next

jax.lax.with_sharding_constraint

Contents

- [`pbroadcast()`](#jax.lax.pbroadcast)

By The JAX authors

© Copyright 2024, The JAX Authors.\
