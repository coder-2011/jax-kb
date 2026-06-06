- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.broadcast_shapes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.broadcast_shapes.rst "Download source file")
-  .pdf

# jax.lax.broadcast_shapes

## Contents

- [`broadcast_shapes()`](#jax.lax.broadcast_shapes)

# jax.lax.broadcast_shapes[\#](#jax-lax-broadcast-shapes "Link to this heading")

jax.lax.broadcast_shapes(*\*shapes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L152-L194)[\#](#jax.lax.broadcast_shapes "Link to this definition")  
Returns the shape that results from NumPy broadcasting of shapes.

This follows the rules of [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Parameters:  
**shapes** – one or more tuples of integers containing the shapes of arrays to be broadcast.

Returns:  
A tuple of integers representing the broadcasted shape.

Raises:  
[**ValueError**](https://docs.python.org/3/library/exceptions.html#ValueError "(in Python v3.14)") – if shapes are not broadcast-compatible.

See also

- [`jax.numpy.broadcast_shapes()`](jax.numpy.broadcast_shapes.html#jax.numpy.broadcast_shapes "jax.numpy.broadcast_shapes"): similar API in the JAX NumPy namespace

Examples

Some examples of broadcasting compatible shapes:

    >>> jnp.broadcast_shapes((1,), (4,))
    (4,)
    >>> jnp.broadcast_shapes((3, 1), (4,))
    (3, 4)
    >>> jnp.broadcast_shapes((3, 1), (1, 4), (5, 1, 1))
    (5, 3, 4)

Error when attempting to broadcast incompatible shapes:

    >>> jnp.broadcast_shapes((3, 1), (4, 1))  
    Traceback (most recent call last):
    ValueError: Incompatible shapes for broadcasting: shapes=[(3, 1), (4, 1)]

[](jax.lax.broadcast_like.html "previous page")

previous

jax.lax.broadcast_like

[](jax.lax.broadcast_to_rank.html "next page")

next

jax.lax.broadcast_to_rank

Contents

- [`broadcast_shapes()`](#jax.lax.broadcast_shapes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
