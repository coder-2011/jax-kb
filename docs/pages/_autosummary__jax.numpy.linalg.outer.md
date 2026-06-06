- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.outer

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.outer.rst "Download source file")
-  .pdf

# jax.numpy.linalg.outer

## Contents

- [`outer()`](#jax.numpy.linalg.outer)

# jax.numpy.linalg.outer[\#](#jax-numpy-linalg-outer "Link to this heading")

jax.numpy.linalg.outer(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1576-L1604)[\#](#jax.numpy.linalg.outer "Link to this definition")  
Compute the outer product of two 1-dimensional arrays.

JAX implementation of [`numpy.linalg.outer()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.outer.html#numpy.linalg.outer "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – array

- **x2** (*ArrayLike*) – array

Returns:  
array containing the outer product of `x1` and `x2`

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

[`jax.numpy.outer()`](jax.numpy.outer.html#jax.numpy.outer "jax.numpy.outer"): similar function in the main [`jax.numpy`](../jax.numpy.html#module-jax.numpy "jax.numpy") module.

Examples

    >>> x1 = jnp.array([1, 2, 3])
    >>> x2 = jnp.array([4, 5, 6])
    >>> jnp.linalg.outer(x1, x2)
    Array([[ 4,  5,  6],
           [ 8, 10, 12],
           [12, 15, 18]], dtype=int32)

[](jax.numpy.linalg.norm.html "previous page")

previous

jax.numpy.linalg.norm

[](jax.numpy.linalg.pinv.html "next page")

next

jax.numpy.linalg.pinv

Contents

- [`outer()`](#jax.numpy.linalg.outer)

By The JAX authors

© Copyright 2024, The JAX Authors.\
