- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.matrix_norm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.matrix_norm.rst "Download source file")
-  .pdf

# jax.numpy.linalg.matrix_norm

## Contents

- [`matrix_norm()`](#jax.numpy.linalg.matrix_norm)

# jax.numpy.linalg.matrix_norm[\#](#jax-numpy-linalg-matrix-norm "Link to this heading")

jax.numpy.linalg.matrix_norm(*x*, */*, *\**, *keepdims=False*, *ord='fro'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1606-L1635)[\#](#jax.numpy.linalg.matrix_norm "Link to this definition")  
Compute the norm of a matrix or stack of matrices.

JAX implementation of [`numpy.linalg.matrix_norm()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_norm.html#numpy.linalg.matrix_norm "(in NumPy v2.4)")

Parameters:  
- **x** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``N)` for which to take the norm.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, keep the reduced dimensions in the output.

- **ord** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – A string or int specifying the type of norm; default is the Frobenius norm. See [`numpy.linalg.norm()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm "(in NumPy v2.4)") for details on available options.

Returns:  
array containing the norm of `x`. Has shape `x.shape[:-2]` if `keepdims` is False, or shape `(...,`` ``1,`` ``1)` if `keepdims` is True.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.vector_norm()`](jax.numpy.linalg.vector_norm.html#jax.numpy.linalg.vector_norm "jax.numpy.linalg.vector_norm"): Norm of a vector or stack of vectors.

- [`jax.numpy.linalg.norm()`](jax.numpy.linalg.norm.html#jax.numpy.linalg.norm "jax.numpy.linalg.norm"): More general matrix or vector norm.

Examples

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6],
    ...                [7, 8, 9]])
    >>> jnp.linalg.matrix_norm(x)
    Array(16.881943, dtype=float32)

[](jax.numpy.linalg.matmul.html "previous page")

previous

jax.numpy.linalg.matmul

[](jax.numpy.linalg.matrix_power.html "next page")

next

jax.numpy.linalg.matrix_power

Contents

- [`matrix_norm()`](#jax.numpy.linalg.matrix_norm)

By The JAX authors

© Copyright 2024, The JAX Authors.\
