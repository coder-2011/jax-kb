- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.broadcast_to

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.broadcast_to.rst "Download source file")
-  .pdf

# jax.experimental.pallas.broadcast_to

## Contents

- [`broadcast_to()`](#jax.experimental.pallas.broadcast_to)

# jax.experimental.pallas.broadcast_to[\#](#jax-experimental-pallas-broadcast-to "Link to this heading")

jax.experimental.pallas.broadcast_to(*a*, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/state/primitives.py#L1069-L1087)[\#](#jax.experimental.pallas.broadcast_to "Link to this definition")  
Broadcasts an array to a new shape.

Parameters:  
- **a** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – The array to broadcast.

- **shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – The desired shape to broadcast to.

Returns:  
An array of shape `shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to")

[](jax.experimental.pallas.empty_like.html "previous page")

previous

jax.experimental.pallas.empty_like

[](jax.experimental.pallas.debug_check.html "next page")

next

jax.experimental.pallas.debug_check

Contents

- [`broadcast_to()`](#jax.experimental.pallas.broadcast_to)

By The JAX authors

© Copyright 2024, The JAX Authors.\
