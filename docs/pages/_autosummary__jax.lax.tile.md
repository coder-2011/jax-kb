- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.tile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.tile.rst "Download source file")
-  .pdf

# jax.lax.tile

## Contents

- [`tile()`](#jax.lax.tile)

# jax.lax.tile[\#](#jax-lax-tile "Link to this heading")

jax.lax.tile(*operand*, *reps*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2940-L2970)[\#](#jax.lax.tile "Link to this definition")  
Tiles an array by repeating it along each dimension.

Parameters:  
- **operand** (*ArrayLike*) – an array to tile.

- **reps** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of integers representing the number of repeats for each dimension. Must have the same length as `operand.ndim`.

Returns:  
A tiled array with shape `(operand.shape[0]`` ``*`` ``reps[0],`` ``...,`` ``operand.shape[-1]`` ``*`` ``reps[-1])`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> x = jnp.array([[1, 2], [3, 4]])
    >>> lax.tile(x, (2, 3))
    Array([[1, 2, 1, 2, 1, 2],
           [3, 4, 3, 4, 3, 4],
           [1, 2, 1, 2, 1, 2],
           [3, 4, 3, 4, 3, 4]], dtype=int32)

    >>> y = jnp.array([1, 2, 3])
    >>> lax.tile(y, (2,))
    Array([1, 2, 3, 1, 2, 3], dtype=int32)

    >>> z = jnp.array([[1], [2]])
    >>> lax.tile(z, (1, 3))
    Array([[1, 1, 1],
           [2, 2, 2]], dtype=int32)

[](jax.lax.tanh.html "previous page")

previous

jax.lax.tanh

[](jax.lax.top_k.html "next page")

next

jax.lax.top_k

Contents

- [`tile()`](#jax.lax.tile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
