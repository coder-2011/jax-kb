- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.hypot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.hypot.rst "Download source file")
-  .pdf

# jax.numpy.hypot

## Contents

- [`hypot()`](#jax.numpy.hypot)

# jax.numpy.hypot[\#](#jax-numpy-hypot "Link to this heading")

jax.numpy.hypot(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3713-L3760)[\#](#jax.numpy.hypot "Link to this definition")  
Return element-wise hypotenuse for the given legs of a right angle triangle.

JAX implementation of [`numpy.hypot`](https://numpy.org/doc/stable/reference/generated/numpy.hypot.html#numpy.hypot "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – scalar or array. Specifies one of the legs of right angle triangle. `complex` dtype are not supported.

- **x2** (*ArrayLike*) – scalar or array. Specifies the other leg of right angle triangle. `complex` dtype are not supported. `x1` and `x2` must either have same shape or be broadcast compatible.

Returns:  
An array containing the hypotenuse for the given given legs `x1` and `x2` of a right angle triangle, promoting to inexact dtype.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.hypot` is a more numerically stable way of computing `jnp.sqrt(x1`` ``**`` ``2`` ``+`` ``x2`` ``**2)`.

Examples

    >>> jnp.hypot(3, 4)
    Array(5., dtype=float32, weak_type=True)
    >>> x1 = jnp.array([[3, -2, 5],
    ...                 [9, 1, -4]])
    >>> x2 = jnp.array([-5, 6, 8])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   jnp.hypot(x1, x2)
    Array([[ 5.831,  6.325,  9.434],
           [10.296,  6.083,  8.944]], dtype=float32)

[](jax.numpy.hstack.html "previous page")

previous

jax.numpy.hstack

[](jax.numpy.i0.html "next page")

next

jax.numpy.i0

Contents

- [`hypot()`](#jax.numpy.hypot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
