- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.array_equiv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.array_equiv.rst "Download source file")
-  .pdf

# jax.numpy.array_equiv

## Contents

- [`array_equiv()`](#jax.numpy.array_equiv)

# jax.numpy.array_equiv[\#](#jax-numpy-array-equiv "Link to this heading")

jax.numpy.array_equiv(*a1*, *a2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5413-L5450)[\#](#jax.numpy.array_equiv "Link to this definition")  
Check if two arrays are element-wise equal.

JAX implementation of [`numpy.array_equiv()`](https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html#numpy.array_equiv "(in NumPy v2.4)").

This function will return `False` if the input arrays cannot be broadcasted to the same shape.

Parameters:  
- **a1** (*ArrayLike*) – first input array to compare.

- **a2** (*ArrayLike*) – second input array to compare.

Returns:  
Boolean scalar array indicating whether the input arrays are element-wise equal after broadcasting.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.allclose()`](jax.numpy.allclose.html#jax.numpy.allclose "jax.numpy.allclose")

- [`jax.numpy.array_equal()`](jax.numpy.array_equal.html#jax.numpy.array_equal "jax.numpy.array_equal")

Examples

    >>> jnp.array_equiv(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    Array(True, dtype=bool)
    >>> jnp.array_equiv(jnp.array([1, 2, 3]), jnp.array([1, 2, 4]))
    Array(False, dtype=bool)
    >>> jnp.array_equiv(jnp.array([[1, 2, 3], [1, 2, 3]]),
    ...                 jnp.array([1, 2, 3]))
    Array(True, dtype=bool)

[](jax.numpy.array_equal.html "previous page")

previous

jax.numpy.array_equal

[](jax.numpy.array_repr.html "next page")

next

jax.numpy.array_repr

Contents

- [`array_equiv()`](#jax.numpy.array_equiv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
