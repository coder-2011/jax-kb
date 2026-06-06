- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.vector_norm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.vector_norm.rst "Download source file")
-  .pdf

# jax.numpy.linalg.vector_norm

## Contents

- [`vector_norm()`](#jax.numpy.linalg.vector_norm)

# jax.numpy.linalg.vector_norm[\#](#jax-numpy-linalg-vector-norm "Link to this heading")

jax.numpy.linalg.vector_norm(*x*, */*, *\**, *axis=None*, *keepdims=False*, *ord=2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L1692-L1758)[\#](#jax.numpy.linalg.vector_norm "Link to this definition")  
Compute the vector norm of a vector or batch of vectors.

JAX implementation of [`numpy.linalg.vector_norm()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html#numpy.linalg.vector_norm "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – N-dimensional array for which to take the norm.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – optional axis along which to compute the vector norm. If None (default) then `x` is flattened and the norm is taken over all values.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, keep the reduced dimensions in the output.

- **ord** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – A string or int specifying the type of norm; default is the 2-norm. See [`numpy.linalg.norm()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm "(in NumPy v2.4)") for details on available options.

Returns:  
array containing the norm of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.matrix_norm()`](jax.numpy.linalg.matrix_norm.html#jax.numpy.linalg.matrix_norm "jax.numpy.linalg.matrix_norm"): Norm of a matrix or stack of matrices.

- [`jax.numpy.linalg.norm()`](jax.numpy.linalg.norm.html#jax.numpy.linalg.norm "jax.numpy.linalg.norm"): More general matrix or vector norm.

Examples

Norm of a single vector:

    >>> x = jnp.array([1., 2., 3.])
    >>> jnp.linalg.vector_norm(x)
    Array(3.7416575, dtype=float32)

Norm of a batch of vectors:

    >>> x = jnp.array([[1., 2., 3.],
    ...                [4., 5., 7.]])
    >>> jnp.linalg.vector_norm(x, axis=1)
    Array([3.7416575, 9.486833 ], dtype=float32)

[](jax.numpy.linalg.trace.html "previous page")

previous

jax.numpy.linalg.trace

[](jax.numpy.linalg.vecdot.html "next page")

next

jax.numpy.linalg.vecdot

Contents

- [`vector_norm()`](#jax.numpy.linalg.vector_norm)

By The JAX authors

© Copyright 2024, The JAX Authors.\
