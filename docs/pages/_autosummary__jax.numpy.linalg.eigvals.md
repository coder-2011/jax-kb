- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.eigvals

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.eigvals.rst "Download source file")
-  .pdf

# jax.numpy.linalg.eigvals

## Contents

- [`eigvals()`](#jax.numpy.linalg.eigvals)

# jax.numpy.linalg.eigvals[\#](#jax-numpy-linalg-eigvals "Link to this heading")

jax.numpy.linalg.eigvals(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L848-L884)[\#](#jax.numpy.linalg.eigvals "Link to this definition")  
Compute the eigenvalues of a general matrix.

JAX implementation of [`numpy.linalg.eigvals()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals "(in NumPy v2.4)").

Parameters:  
**a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``M)` for which to compute the eigenvalues.

Returns:  
An array of shape `(...,`` ``M)` containing the eigenvalues.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.eig()`](jax.numpy.linalg.eig.html#jax.numpy.linalg.eig "jax.numpy.linalg.eig"): computes eigenvalues eigenvectors of a general matrix.

- [`jax.numpy.linalg.eigh()`](jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh "jax.numpy.linalg.eigh"): computes eigenvalues eigenvectors of a Hermitian matrix.

Notes

- This differs from [`numpy.linalg.eigvals()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html#numpy.linalg.eigvals "(in NumPy v2.4)") in that the return type of [`jax.numpy.linalg.eigvals()`](#jax.numpy.linalg.eigvals "jax.numpy.linalg.eigvals") is always complex64 for 32-bit input, and complex128 for 64-bit input.

- At present, non-symmetric eigendecomposition is only implemented on the CPU backend.

Examples

    >>> a = jnp.array([[1., 2.],
    ...                [2., 1.]])
    >>> w = jnp.linalg.eigvals(a)
    >>> with jnp.printoptions(precision=2):
    ...  w
    Array([ 3.+0.j, -1.+0.j], dtype=complex64)

[](jax.numpy.linalg.eigh.html "previous page")

previous

jax.numpy.linalg.eigh

[](jax.numpy.linalg.eigvalsh.html "next page")

next

jax.numpy.linalg.eigvalsh

Contents

- [`eigvals()`](#jax.numpy.linalg.eigvals)

By The JAX authors

© Copyright 2024, The JAX Authors.\
