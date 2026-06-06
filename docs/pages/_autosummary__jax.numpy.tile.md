- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.tile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.tile.rst "Download source file")
-  .pdf

# jax.numpy.tile

## Contents

- [`tile()`](#jax.numpy.tile)

# jax.numpy.tile[\#](#jax-numpy-tile "Link to this heading")

jax.numpy.tile(*A*, *reps*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4459-L4505)[\#](#jax.numpy.tile "Link to this definition")  
Construct an array by repeating `A` along specified dimensions.

JAX implementation of [`numpy.tile()`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html#numpy.tile "(in NumPy v2.4)").

If `A` is an array of shape `(d1,`` ``d2,`` ``...,`` ``dn)` and `reps` is a sequence of integers, the resulting array will have a shape of `(reps[0]`` ``*`` ``d1,`` ``reps[1]`` ``*`` ``d2,`` ``...,`` ``reps[n]`` ``*`` ``dn)`, with `A` tiled along each dimension.

Parameters:  
- **A** (*ArrayLike*) – input array to be repeated. Can be of any shape or dimension.

- **reps** (*DimSize* *\|* *Sequence\[DimSize\]*) – specifies the number of repetitions along each axis.

Returns:  
a new array where the input array has been repeated according to `reps`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.repeat()`](jax.numpy.repeat.html#jax.numpy.repeat "jax.numpy.repeat"): Construct an array from repeated elements.

- [`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to"): Broadcast an array to a specified shape.

Examples

    >>> arr = jnp.array([1, 2])
    >>> jnp.tile(arr, 2)
    Array([1, 2, 1, 2], dtype=int32)
    >>> arr = jnp.array([[1, 2],
    ...                  [3, 4,]])
    >>> jnp.tile(arr, (2, 1))
    Array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]], dtype=int32)

[](jax.numpy.tensordot.html "previous page")

previous

jax.numpy.tensordot

[](jax.numpy.trace.html "next page")

next

jax.numpy.trace

Contents

- [`tile()`](#jax.numpy.tile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
