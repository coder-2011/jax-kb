- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.broadcast_in_dim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.broadcast_in_dim.rst "Download source file")
-  .pdf

# jax.lax.broadcast_in_dim

## Contents

- [`broadcast_in_dim()`](#jax.lax.broadcast_in_dim)

# jax.lax.broadcast_in_dim[\#](#jax-lax-broadcast-in-dim "Link to this heading")

jax.lax.broadcast_in_dim(*operand*, *shape*, *broadcast_dimensions*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2865-L2931)[\#](#jax.lax.broadcast_in_dim "Link to this definition")  
General broadcasting operation.

This function lowers directly to the [stablehlo.broadcast_in_dim](https://openxla.org/stablehlo/spec#broadcast_in_dim) operation.

Parameters:  
- **operand** (*ArrayLike*) – an array

- **shape** (*Shape*) – the shape of the target array

- **broadcast_dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – to which dimension in the target shape each dimension of the operand shape corresponds to. That is, dimension i of the operand becomes dimension broadcast_dimensions\[i\] of the result.

Returns:  
An array containing the result.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.broadcast()`](jax.lax.broadcast.html#jax.lax.broadcast "jax.lax.broadcast"): simpler interface to add new leading dimensions.

- [`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to"): NumPy-style API for general broadcasting.

Examples

Here is an example of implementing simple NumPy-style broadcasting:

    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> import numpy as np

    >>> arr = jnp.arange(3).reshape(3, 1)
    >>> target_shape = (2, 3, 4)
    >>> result = lax.broadcast_in_dim(arr, target_shape, broadcast_dimensions=(1, 2))
    >>> result.shape
    (2, 3, 4)

The above is equivalent to [`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to"):

    >>> result_jnp = jnp.broadcast_to(result, target_shape)
    >>> np.testing.assert_array_equal(result, result_jnp)

However, [`broadcast_in_dim()`](#jax.lax.broadcast_in_dim "jax.lax.broadcast_in_dim") is more general, allowing implicit transposes as part of the single broadcasting operation:

    >>> result = lax.broadcast_in_dim(arr, target_shape, broadcast_dimensions=(1, 0))
    >>> result.shape
    (2, 3, 4)

This more general operation has no direct equivlant in the NumPy-style broadcasting API, but can be replicated by appropriately adding and transposing input dimensions:

    >>> result_jnp = jnp.broadcast_to(jnp.expand_dims(arr, 0).transpose(), target_shape)
    >>> np.testing.assert_array_equal(result, result_jnp)

[](jax.lax.broadcast.html "previous page")

previous

jax.lax.broadcast

[](jax.lax.broadcast_like.html "next page")

next

jax.lax.broadcast_like

Contents

- [`broadcast_in_dim()`](#jax.lax.broadcast_in_dim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
