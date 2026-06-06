- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.divmod

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.divmod.rst "Download source file")
-  .pdf

# jax.numpy.divmod

## Contents

- [`divmod()`](#jax.numpy.divmod)

# jax.numpy.divmod[\#](#jax-numpy-divmod "Link to this heading")

jax.numpy.divmod(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2545-L2587)[\#](#jax.numpy.divmod "Link to this definition")  
Calculates the integer quotient and remainder of x1 by x2 element-wise

JAX implementation of [`numpy.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html#numpy.divmod "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – Input array, the dividend

- **x2** (*ArrayLike*) – Input array, the divisor

Returns:  
A tuple of arrays `(x1`` ``//`` ``x2,`` ``x1`` ``%`` ``x2)`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.floor_divide()`](jax.numpy.floor_divide.html#jax.numpy.floor_divide "jax.numpy.floor_divide"): floor division function

- [`jax.numpy.remainder()`](jax.numpy.remainder.html#jax.numpy.remainder "jax.numpy.remainder"): remainder function

Examples

    >>> x1 = jnp.array([10, 20, 30])
    >>> x2 = jnp.array([3, 4, 7])
    >>> jnp.divmod(x1, x2)
    (Array([3, 5, 4], dtype=int32), Array([1, 0, 2], dtype=int32))

    >>> x1 = jnp.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    >>> x2 = 3
    >>> jnp.divmod(x1, x2)
    (Array([-2, -2, -1, -1, -1,  0,  0,  0,  1,  1,  1], dtype=int32),
     Array([1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int32))

    >>> x1 = jnp.array([6, 6, 6], dtype=jnp.int32)
    >>> x2 = jnp.array([1.9, 2.5, 3.1], dtype=jnp.float32)
    >>> jnp.divmod(x1, x2)
    (Array([3., 2., 1.], dtype=float32),
     Array([0.30000007, 1.        , 2.9       ], dtype=float32))

[](jax.numpy.divide.html "previous page")

previous

jax.numpy.divide

[](jax.numpy.dot.html "next page")

next

jax.numpy.dot

Contents

- [`divmod()`](#jax.numpy.divmod)

By The JAX authors

© Copyright 2024, The JAX Authors.\
