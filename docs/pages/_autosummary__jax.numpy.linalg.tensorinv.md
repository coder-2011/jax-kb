- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.tensorinv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.tensorinv.rst "Download source file")
-  .pdf

# jax.numpy.linalg.tensorinv

## Contents

- [`tensorinv()`](#jax.numpy.linalg.tensorinv)

# jax.numpy.linalg.tensorinv[\#](#jax-numpy-linalg-tensorinv "Link to this heading")

jax.numpy.linalg.tensorinv(*a*, *ind=2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L2021-L2061)[\#](#jax.numpy.linalg.tensorinv "Link to this definition")  
Compute the tensor inverse of an array.

JAX implementation of [`numpy.linalg.tensorinv()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorinv.html#numpy.linalg.tensorinv "(in NumPy v2.4)").

This computes the inverse of the [`tensordot()`](jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot") operation with the same `ind` value.

Parameters:  
- **a** (*ArrayLike*) – array to be inverted. Must have `prod(a.shape[:ind])`` ``==`` ``prod(a.shape[ind:])`

- **ind** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – positive integer specifying the number of indices in the tensor product.

Returns:  
array of shape `(*a.shape[ind:],`` ``*a.shape[:ind])` containing the tensor inverse of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.tensordot()`](jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot")

- [`jax.numpy.linalg.tensorsolve()`](jax.numpy.linalg.tensorsolve.html#jax.numpy.linalg.tensorsolve "jax.numpy.linalg.tensorsolve")

Examples

    >>> key = jax.random.key(1337)
    >>> x = jax.random.normal(key, shape=(2, 2, 4))
    >>> xinv = jnp.linalg.tensorinv(x, 2)
    >>> xinv_x = jnp.linalg.tensordot(xinv, x, axes=2)
    >>> jnp.allclose(xinv_x, jnp.eye(4), atol=1E-4)
    Array(True, dtype=bool)

[](jax.numpy.linalg.tensordot.html "previous page")

previous

jax.numpy.linalg.tensordot

[](jax.numpy.linalg.tensorsolve.html "next page")

next

jax.numpy.linalg.tensorsolve

Contents

- [`tensorinv()`](#jax.numpy.linalg.tensorinv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
