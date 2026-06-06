- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.allclose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.allclose.rst "Download source file")
-  .pdf

# jax.numpy.allclose

## Contents

- [`allclose()`](#jax.numpy.allclose)

# jax.numpy.allclose[\#](#jax-numpy-allclose "Link to this heading")

jax.numpy.allclose(*a*, *b*, *rtol=1e-05*, *atol=1e-08*, *equal_nan=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3561-L3608)[\#](#jax.numpy.allclose "Link to this definition")  
Check if two arrays are element-wise approximately equal within a tolerance.

JAX implementation of [`numpy.allclose()`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html#numpy.allclose "(in NumPy v2.4)").

Essentially this function evaluates the following condition:

\\\|a - b\| \le \mathtt{atol} + \mathtt{rtol} \* \|b\|\\

`jnp.inf` in `a` will be considered equal to `jnp.inf` in `b`.

Parameters:  
- **a** (*ArrayLike*) – first input array to compare.

- **b** (*ArrayLike*) – second input array to compare.

- **rtol** (*ArrayLike*) – relative tolerance used for approximate equality. Default = 1e-05.

- **atol** (*ArrayLike*) – absolute tolerance used for approximate equality. Default = 1e-08.

- **equal_nan** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Boolean. If `True`, NaNs in `a` will be considered equal to NaNs in `b`. Default is `False`.

Returns:  
Boolean scalar array indicating whether the input arrays are element-wise approximately equal within the specified tolerances.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.isclose()`](jax.numpy.isclose.html#jax.numpy.isclose "jax.numpy.isclose")

- [`jax.numpy.equal()`](jax.numpy.equal.html#jax.numpy.equal "jax.numpy.equal")

Examples

    >>> jnp.allclose(jnp.array([1e6, 2e6, 3e6]), jnp.array([1e6, 2e6, 3e7]))
    Array(False, dtype=bool)
    >>> jnp.allclose(jnp.array([1e6, 2e6, 3e6]),
    ...              jnp.array([1.00008e6, 2.00008e7, 3.00008e8]), rtol=1e3)
    Array(True, dtype=bool)
    >>> jnp.allclose(jnp.array([1e6, 2e6, 3e6]),
    ...              jnp.array([1.00001e6, 2.00002e6, 3.00009e6]), atol=1e3)
    Array(True, dtype=bool)
    >>> jnp.allclose(jnp.array([jnp.nan, 1, 2]),
    ...              jnp.array([jnp.nan, 1, 2]), equal_nan=True)
    Array(True, dtype=bool)

[](jax.numpy.all.html "previous page")

previous

jax.numpy.all

[](jax.numpy.amax.html "next page")

next

jax.numpy.amax

Contents

- [`allclose()`](#jax.numpy.allclose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
