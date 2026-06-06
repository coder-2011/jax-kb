- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.modf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.modf.rst "Download source file")
-  .pdf

# jax.numpy.modf

## Contents

- [`modf()`](#jax.numpy.modf)

# jax.numpy.modf[\#](#jax-numpy-modf "Link to this heading")

jax.numpy.modf(*x*, */*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L3426-L3458)[\#](#jax.numpy.modf "Link to this definition")  
Return element-wise fractional and integral parts of the input array.

JAX implementation of [`numpy.modf`](https://numpy.org/doc/stable/reference/generated/numpy.modf.html#numpy.modf "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – input array or scalar.

- **out** – Not used by JAX.

Returns:  
An array containing the fractional and integral parts of the elements of `x`, promoting dtypes inexact.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.numpy.divmod()`](jax.numpy.divmod.html#jax.numpy.divmod "jax.numpy.divmod"): Calculates the integer quotient and remainder of `x1` by `x2` element-wise.

Examples

    >>> jnp.modf(4.8)
    (Array(0.8000002, dtype=float32, weak_type=True), Array(4., dtype=float32, weak_type=True))
    >>> x = jnp.array([-3.4, -5.7, 0.6, 1.5, 2.3])
    >>> jnp.modf(x)
    (Array([-0.4000001 , -0.6999998 ,  0.6       ,  0.5       ,  0.29999995],      dtype=float32), Array([-3., -5.,  0.,  1.,  2.], dtype=float32))

[](jax.numpy.mod.html "previous page")

previous

jax.numpy.mod

[](jax.numpy.moveaxis.html "next page")

next

jax.numpy.moveaxis

Contents

- [`modf()`](#jax.numpy.modf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
