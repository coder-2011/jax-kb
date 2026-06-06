- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.array_equal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.array_equal.rst "Download source file")
-  .pdf

# jax.numpy.array_equal

## Contents

- [`array_equal()`](#jax.numpy.array_equal)

# jax.numpy.array_equal[\#](#jax-numpy-array-equal "Link to this heading")

jax.numpy.array_equal(*a1*, *a2*, *equal_nan=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5371-L5411)[\#](#jax.numpy.array_equal "Link to this definition")  
Check if two arrays are element-wise equal.

JAX implementation of [`numpy.array_equal()`](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html#numpy.array_equal "(in NumPy v2.4)").

Parameters:  
- **a1** (*ArrayLike*) – first input array to compare.

- **a2** (*ArrayLike*) – second input array to compare.

- **equal_nan** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Boolean. If `True`, NaNs in `a1` will be considered equal to NaNs in `a2`. Default is `False`.

Returns:  
Boolean scalar array indicating whether the input arrays are element-wise equal.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.allclose()`](jax.numpy.allclose.html#jax.numpy.allclose "jax.numpy.allclose")

- [`jax.numpy.array_equiv()`](jax.numpy.array_equiv.html#jax.numpy.array_equiv "jax.numpy.array_equiv")

Examples

    >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    Array(True, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2]))
    Array(False, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))
    Array(False, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, float('nan')]),
    ...                 jnp.array([1, 2, float('nan')]))
    Array(False, dtype=bool)
    >>> jnp.array_equal(jnp.array([1, 2, float('nan')]),
    ...                 jnp.array([1, 2, float('nan')]), equal_nan=True)
    Array(True, dtype=bool)

[](jax.numpy.array.html "previous page")

previous

jax.numpy.array

[](jax.numpy.array_equiv.html "next page")

next

jax.numpy.array_equiv

Contents

- [`array_equal()`](#jax.numpy.array_equal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
