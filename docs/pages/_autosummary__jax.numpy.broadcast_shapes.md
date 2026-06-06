- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.broadcast_shapes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.broadcast_shapes.rst "Download source file")
-  .pdf

# jax.numpy.broadcast_shapes

## Contents

- [`broadcast_shapes()`](#jax.numpy.broadcast_shapes)

# jax.numpy.broadcast_shapes[\#](#jax-numpy-broadcast-shapes "Link to this heading")

jax.numpy.broadcast_shapes(*\*shapes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3002-L3041)[\#](#jax.numpy.broadcast_shapes "Link to this definition")  
Broadcast input shapes to a common output shape.

JAX implementation of [`numpy.broadcast_shapes()`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_shapes.html#numpy.broadcast_shapes "(in NumPy v2.4)"). JAX uses NumPy-style broadcasting rules, which you can read more about at [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Parameters:  
**shapes** – 0 or more shapes specified as sequences of integers

Returns:  
The broadcasted shape as a tuple of integers.

See also

- [`jax.numpy.broadcast_arrays()`](jax.numpy.broadcast_arrays.html#jax.numpy.broadcast_arrays "jax.numpy.broadcast_arrays"): broadcast arrays to a common shape.

- [`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to"): broadcast an array to a specified shape.

Examples

Some compatible shapes:

    >>> jnp.broadcast_shapes((1,), (4,))
    (4,)
    >>> jnp.broadcast_shapes((3, 1), (4,))
    (3, 4)
    >>> jnp.broadcast_shapes((3, 1), (1, 4), (5, 1, 1))
    (5, 3, 4)

Incompatible shapes:

    >>> jnp.broadcast_shapes((3, 1), (4, 1))  
    Traceback (most recent call last):
    ValueError: Incompatible shapes for broadcasting: shapes=[(3, 1), (4, 1)]

[](jax.numpy.broadcast_arrays.html "previous page")

previous

jax.numpy.broadcast_arrays

[](jax.numpy.broadcast_to.html "next page")

next

jax.numpy.broadcast_to

Contents

- [`broadcast_shapes()`](#jax.numpy.broadcast_shapes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
