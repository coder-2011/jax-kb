- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.broadcast_arrays

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.broadcast_arrays.rst "Download source file")
-  .pdf

# jax.numpy.broadcast_arrays

## Contents

- [`broadcast_arrays()`](#jax.numpy.broadcast_arrays)

# jax.numpy.broadcast_arrays[\#](#jax-numpy-broadcast-arrays "Link to this heading")

jax.numpy.broadcast_arrays(*\*args*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3043-L3082)[\#](#jax.numpy.broadcast_arrays "Link to this definition")  
Broadcast arrays to a common shape.

JAX implementation of [`numpy.broadcast_arrays()`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays "(in NumPy v2.4)"). JAX uses NumPy-style broadcasting rules, which you can read more about at [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Parameters:  
**args** (*ArrayLike*) – zero or more array-like objects to be broadcasted.

Returns:  
a list of arrays containing broadcasted copies of the inputs.

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.broadcast_shapes()`](jax.numpy.broadcast_shapes.html#jax.numpy.broadcast_shapes "jax.numpy.broadcast_shapes"): broadcast input shapes to a common shape.

- [`jax.numpy.broadcast_to()`](jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to"): broadcast an array to a specified shape.

Examples

    >>> x = jnp.arange(3)
    >>> y = jnp.int32(1)
    >>> jnp.broadcast_arrays(x, y)
    [Array([0, 1, 2], dtype=int32), Array([1, 1, 1], dtype=int32)]

    >>> x = jnp.array([[1, 2, 3]])
    >>> y = jnp.array([[10],
    ...                [20]])
    >>> x2, y2 = jnp.broadcast_arrays(x, y)
    >>> x2
    Array([[1, 2, 3],
           [1, 2, 3]], dtype=int32)
    >>> y2
    Array([[10, 10, 10],
           [20, 20, 20]], dtype=int32)

[](jax.numpy.bool_.html "previous page")

previous

jax.numpy.bool\_

[](jax.numpy.broadcast_shapes.html "next page")

next

jax.numpy.broadcast_shapes

Contents

- [`broadcast_arrays()`](#jax.numpy.broadcast_arrays)

By The JAX authors

© Copyright 2024, The JAX Authors.\
