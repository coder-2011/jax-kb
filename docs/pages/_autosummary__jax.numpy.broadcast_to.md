- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.broadcast_to

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.broadcast_to.rst "Download source file")
-  .pdf

# jax.numpy.broadcast_to

## Contents

- [`broadcast_to()`](#jax.numpy.broadcast_to)

# jax.numpy.broadcast_to[\#](#jax-numpy-broadcast-to "Link to this heading")

jax.numpy.broadcast_to(*array*, *shape*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3084-L3121)[\#](#jax.numpy.broadcast_to "Link to this definition")  
Broadcast an array to a specified shape.

JAX implementation of [`numpy.broadcast_to()`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to "(in NumPy v2.4)"). JAX uses NumPy-style broadcasting rules, which you can read more about at [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Parameters:  
- **array** (*ArrayLike*) – array to be broadcast.

- **shape** (*DimSize* *\|* *Shape*) – shape to which the array will be broadcast.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
a copy of array broadcast to the specified shape.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.broadcast_arrays()`](jax.numpy.broadcast_arrays.html#jax.numpy.broadcast_arrays "jax.numpy.broadcast_arrays"): broadcast arrays to a common shape.

- [`jax.numpy.broadcast_shapes()`](jax.numpy.broadcast_shapes.html#jax.numpy.broadcast_shapes "jax.numpy.broadcast_shapes"): broadcast input shapes to a common shape.

Examples

    >>> x = jnp.int32(1)
    >>> jnp.broadcast_to(x, (1, 4))
    Array([[1, 1, 1, 1]], dtype=int32)

    >>> x = jnp.array([1, 2, 3])
    >>> jnp.broadcast_to(x, (2, 3))
    Array([[1, 2, 3],
           [1, 2, 3]], dtype=int32)

    >>> x = jnp.array([[2], [4]])
    >>> jnp.broadcast_to(x, (2, 4))
    Array([[2, 2, 2, 2],
           [4, 4, 4, 4]], dtype=int32)

[](jax.numpy.broadcast_shapes.html "previous page")

previous

jax.numpy.broadcast_shapes

[](jax.numpy.c_.html "next page")

next

jax.numpy.c\_

Contents

- [`broadcast_to()`](#jax.numpy.broadcast_to)

By The JAX authors

© Copyright 2024, The JAX Authors.\
