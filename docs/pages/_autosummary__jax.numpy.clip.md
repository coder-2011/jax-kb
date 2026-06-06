- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.clip

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.clip.rst "Download source file")
-  .pdf

# jax.numpy.clip

## Contents

- [`clip()`](#jax.numpy.clip)

# jax.numpy.clip[\#](#jax-numpy-clip "Link to this heading")

jax.numpy.clip(*arr=None*, */*, *min=None*, *max=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3361-L3411)[\#](#jax.numpy.clip "Link to this definition")  
Clip array values to a specified range.

JAX implementation of [`numpy.clip()`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html#numpy.clip "(in NumPy v2.4)").

Parameters:  
- **arr** (*ArrayLike* *\|* *None*) – N-dimensional array to be clipped.

- **min** (*ArrayLike* *\|* *None*) – optional minimum value of the clipped range; if `None` (default) then result will not be clipped to any minimum value. If specified, it should be broadcast-compatible with `arr` and `max`.

- **max** (*ArrayLike* *\|* *None*) – optional maximum value of the clipped range; if `None` (default) then result will not be clipped to any maximum value. If specified, it should be broadcast-compatible with `arr` and `min`.

Returns:  
An array containing values from `arr`, with values smaller than `min` set to `min`, and values larger than `max` set to `max`. Wherever `min` is larger than `max`, the value of `max` is returned.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.minimum()`](jax.numpy.minimum.html#jax.numpy.minimum "jax.numpy.minimum"): Compute the element-wise minimum value of two arrays.

- [`jax.numpy.maximum()`](jax.numpy.maximum.html#jax.numpy.maximum "jax.numpy.maximum"): Compute the element-wise maximum value of two arrays.

Examples

    >>> arr = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])
    >>> jnp.clip(arr, 2, 5)
    Array([2, 2, 2, 3, 4, 5, 5, 5], dtype=int32)

[](jax.numpy.choose.html "previous page")

previous

jax.numpy.choose

[](jax.numpy.column_stack.html "next page")

next

jax.numpy.column_stack

Contents

- [`clip()`](#jax.numpy.clip)

By The JAX authors

© Copyright 2024, The JAX Authors.\
