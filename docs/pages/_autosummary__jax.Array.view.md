- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.view

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.view.rst "Download source file")
-  .pdf

# jax.Array.view

## Contents

- [`Array.view()`](#jax.Array.view)

# jax.Array.view[\#](#jax-array-view "Link to this heading")

*abstract* Array.view(*dtype=None*, *type=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_methods.py#L547-L645)[\#](#jax.Array.view "Link to this definition")  
Return a bitwise copy of the array, viewed as a new dtype.

This is fuller-featured wrapper around [`jax.lax.bitcast_convert_type()`](jax.lax.bitcast_convert_type.html#jax.lax.bitcast_convert_type "jax.lax.bitcast_convert_type").

If the source and target dtype have the same bitwidth, the result has the same shape as the input array. If the bitwidth of the target dtype is different from the source, the size of the last axis of the result is adjusted accordingly.

    >>> jnp.zeros([1,2,3], dtype=jnp.int16).view(jnp.int8).shape
    (1, 2, 6)
    >>> jnp.zeros([1,2,4], dtype=jnp.int8).view(jnp.int16).shape
    (1, 2, 2)

Conversions involving booleans are not well-defined in all situations. With regards to the shape of result as explained above, booleans are treated as having a bitwidth of 8. However, when converting to a boolean array, the input should only contain 0 or 1 bytes. Otherwise, results may be unpredictable or may change depending on how the result is used.

This conversion is guaranteed and safe:

    >>> jnp.array([1, 0, 1], dtype=jnp.int8).view(jnp.bool_)
    Array([ True, False,  True], dtype=bool)

However, there are no guarantees about the results of any expression involving a view such as this: `jnp.array([1,`` ``2,`` ``3],`` ``dtype=jnp.int8).view(jnp.bool_)`. In particular, the results may change between JAX releases and depending on the platform. To safely convert such an array to a boolean array, compare it with 0:

    >>> jnp.array([1, 2, 0], dtype=jnp.int8) != 0
    Array([ True,  True, False], dtype=bool)

Parameters:  
- **dtype** (*DTypeLike* *\|* *None*) – An optional output dtype. If not specified, the output dtype is the same as the input dtype.

- **type** (*None*) – Not implemented; accepted for NumPy compatibility.

- **self** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

Returns:  
The array, viewed as the new dtype. Unlike NumPy, the array may or may not be a copy of the input array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.Array.var.html "previous page")

previous

jax.Array.var

[](jax.Array.T.html "next page")

next

jax.Array.T

Contents

- [`Array.view()`](#jax.Array.view)

By The JAX authors

© Copyright 2024, The JAX Authors.\
