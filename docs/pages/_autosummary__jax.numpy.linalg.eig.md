- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.eig

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.eig.rst "Download source file")
-  .pdf

# jax.numpy.linalg.eig

## Contents

- [`eig()`](#jax.numpy.linalg.eig)

# jax.numpy.linalg.eig[\#](#jax-numpy-linalg-eig "Link to this heading")

jax.numpy.linalg.eig(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L798-L846)[\#](#jax.numpy.linalg.eig "Link to this definition")  
Compute the eigenvalues and eigenvectors of a square array.

JAX implementation of [`numpy.linalg.eig()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig "(in NumPy v2.4)").

Parameters:  
**a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``M)` for which to compute the eigenvalues and vectors.

Returns:  
- `eigenvalues`: an array of shape `(...,`` ``M)` containing the eigenvalues.

- `eigenvectors`: an array of shape `(...,`` ``M,`` ``M)`, where column `v[:,`` ``i]` is the eigenvector corresponding to the eigenvalue `w[i]`.

Return type:  
A namedtuple `(eigenvalues,`` ``eigenvectors)`. The namedtuple has fields

Notes

- This differs from [`numpy.linalg.eig()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html#numpy.linalg.eig "(in NumPy v2.4)") in that the return type of [`jax.numpy.linalg.eig()`](#jax.numpy.linalg.eig "jax.numpy.linalg.eig") is always complex64 for 32-bit input, and complex128 for 64-bit input.

- At present, non-symmetric eigendecomposition is only implemented on the CPU and GPU backends. For more details about the GPU implementation, see the documentation for [`jax.lax.linalg.eig()`](jax.lax.linalg.eig.html#jax.lax.linalg.eig "jax.lax.linalg.eig").

- Currently autodiff is not supported for computation of non-symmetric eigenvectors; see [jax-ml/jax#2748](https://github.com/jax-ml/jax/issues/2748).

See also

- [`jax.lax.linalg.eig()`](jax.lax.linalg.eig.html#jax.lax.linalg.eig "jax.lax.linalg.eig"): similar function with different eigenvector options and device-specific implementations.

- [`jax.numpy.linalg.eigh()`](jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh "jax.numpy.linalg.eigh"): eigenvectors and eigenvalues of a Hermitian matrix.

- [`jax.numpy.linalg.eigvals()`](jax.numpy.linalg.eigvals.html#jax.numpy.linalg.eigvals "jax.numpy.linalg.eigvals"): compute eigenvalues only.

Examples

    >>> a = jnp.array([[1., 2.],
    ...                [2., 1.]])
    >>> w, v = jnp.linalg.eig(a)
    >>> with jax.numpy.printoptions(precision=4):
    ...   w
    Array([ 3.+0.j, -1.+0.j], dtype=complex64)
    >>> v
    Array([[ 0.70710677+0.j, -0.70710677+0.j],
           [ 0.70710677+0.j,  0.70710677+0.j]], dtype=complex64)

[](jax.numpy.linalg.diagonal.html "previous page")

previous

jax.numpy.linalg.diagonal

[](jax.numpy.linalg.eigh.html "next page")

next

jax.numpy.linalg.eigh

Contents

- [`eig()`](#jax.numpy.linalg.eig)

By The JAX authors

© Copyright 2024, The JAX Authors.\
