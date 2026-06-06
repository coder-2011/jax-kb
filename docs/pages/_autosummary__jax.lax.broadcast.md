- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.broadcast

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.broadcast.rst "Download source file")
-  .pdf

# jax.lax.broadcast

## Contents

- [`broadcast()`](#jax.lax.broadcast)

# jax.lax.broadcast[\#](#jax-lax-broadcast "Link to this heading")

jax.lax.broadcast(*operand*, *sizes*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2834-L2864)[\#](#jax.lax.broadcast "Link to this definition")  
Broadcasts an array, adding new leading dimensions only.

Parameters:  
- **operand** (*ArrayLike*) – an array

- **sizes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of integers, giving the sizes of new leading dimensions to add to the front of the array.

Returns:  
The result array, of shape `(*sizes,`` ``*operand.shape)` containing broadcasted values of `operand`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.broadcast_in_dim()`](jax.lax.broadcast_in_dim.html#jax.lax.broadcast_in_dim "jax.lax.broadcast_in_dim"): general broadcasting at any dimension in the array.

- [`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to"): NumPy-style API for general broadcasting.

Examples

    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> arr = jnp.zeros((4, 5))
    >>> result = lax.broadcast(arr, (2, 3))
    >>> result.shape
    (2, 3, 4, 5)

[](jax.lax.population_count.html "previous page")

previous

jax.lax.population_count

[](jax.lax.broadcast_in_dim.html "next page")

next

jax.lax.broadcast_in_dim

Contents

- [`broadcast()`](#jax.lax.broadcast)

By The JAX authors

© Copyright 2024, The JAX Authors.\
