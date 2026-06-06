- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ptp

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ptp.rst "Download source file")
-  .pdf

# jax.numpy.ptp

## Contents

- [`ptp()`](#jax.numpy.ptp)

# jax.numpy.ptp[\#](#jax-numpy-ptp "Link to this heading")

jax.numpy.ptp(*a*, *axis=None*, *out=None*, *keepdims=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L1284-L1325)[\#](#jax.numpy.ptp "Link to this definition")  
Return the peak-to-peak range along a given axis.

JAX implementation of [`numpy.ptp()`](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html#numpy.ptp "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **axis** (*Axis*) – optional, int or sequence of ints, default=None. Axis along which the range is computed. If None, the range is computed on the flattened array.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **out** (*None*) – Unused by JAX.

Returns:  
An array with the range of elements along specified axis of input.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

By default, `jnp.ptp` computes the range along all axes.

    >>> x = jnp.array([[1, 3, 5, 2],
    ...                [4, 6, 8, 1],
    ...                [7, 9, 3, 4]])
    >>> jnp.ptp(x)
    Array(8, dtype=int32)

If `axis=1`, computes the range along axis 1.

    >>> jnp.ptp(x, axis=1)
    Array([4, 7, 6], dtype=int32)

To preserve the dimensions of input, you can set `keepdims=True`.

    >>> jnp.ptp(x, axis=1, keepdims=True)
    Array([[4],
           [7],
           [6]], dtype=int32)

[](jax.numpy.promote_types.html "previous page")

previous

jax.numpy.promote_types

[](jax.numpy.put.html "next page")

next

jax.numpy.put

Contents

- [`ptp()`](#jax.numpy.ptp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
