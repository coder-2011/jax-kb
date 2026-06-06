- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.broadcast_like

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.broadcast_like.rst "Download source file")
-  .pdf

# jax.lax.broadcast_like

## Contents

- [`broadcast_like()`](#jax.lax.broadcast_like)

# jax.lax.broadcast_like[\#](#jax-lax-broadcast-like "Link to this heading")

jax.lax.broadcast_like(*arr*, *like_arr*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2805-L2832)[\#](#jax.lax.broadcast_like "Link to this definition")  
Broadcasts an array to match the shape and sharding of another array.

Parameters:  
- **arr** (*ArrayLike*) – an array to be broadcasted.

- **like_arr** (*ArrayLike*) – an array whose shape and sharding should be matched.

Returns:  
An array containing the broadcasted values of `arr`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.broadcast()`](jax.lax.broadcast.html#jax.lax.broadcast "jax.lax.broadcast"): simpler interface to add new leading dimensions.

- [`jax.lax.broadcast_in_dim()`](jax.lax.broadcast_in_dim.html#jax.lax.broadcast_in_dim "jax.lax.broadcast_in_dim"): general broadcasting at any dimension in the array.

- [`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to"): NumPy-style API for general broadcasting.

Examples

    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> arr = jnp.array([1, 2, 3])
    >>> like_arr = jnp.zeros((2, 3))
    >>> lax.broadcast_like(arr, like_arr)
    Array([[1, 2, 3],
           [1, 2, 3]], dtype=int32)

[](jax.lax.broadcast_in_dim.html "previous page")

previous

jax.lax.broadcast_in_dim

[](jax.lax.broadcast_shapes.html "next page")

next

jax.lax.broadcast_shapes

Contents

- [`broadcast_like()`](#jax.lax.broadcast_like)

By The JAX authors

© Copyright 2024, The JAX Authors.\
