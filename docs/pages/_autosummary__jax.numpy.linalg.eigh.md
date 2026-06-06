- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.eigh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.eigh.rst "Download source file")
-  .pdf

# jax.numpy.linalg.eigh

## Contents

- [`eigh()`](#jax.numpy.linalg.eigh)

# jax.numpy.linalg.eigh[\#](#jax-numpy-linalg-eigh "Link to this heading")

jax.numpy.linalg.eigh(*a*, *UPLO=None*, *symmetrize_input=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L886-L942)[\#](#jax.numpy.linalg.eigh "Link to this definition")  
Compute the eigenvalues and eigenvectors of a Hermitian matrix.

JAX implementation of [`numpy.linalg.eigh()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html#numpy.linalg.eigh "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``M)`, containing the Hermitian (if complex) or symmetric (if real) matrix.

- **UPLO** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – specifies whether the calculation is done with the lower triangular part of `a` (`'L'`, default) or the upper triangular part (`'U'`).

- **symmetrize_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default) then input is symmetrized, which leads to better behavior under automatic differentiation. Note that when this is set to True, both the upper and lower triangles of the input will be used in computing the decomposition.

Returns:  
A namedtuple `(eigenvalues,`` ``eigenvectors)` where

- `eigenvalues`: an array of shape `(...,`` ``M)` containing the eigenvalues, sorted in ascending order.

- `eigenvectors`: an array of shape `(...,`` ``M,`` ``M)`, where column `v[:,`` ``i]` is the normalized eigenvector corresponding to the eigenvalue `w[i]`.

Return type:  
EighResult

See also

- [`jax.numpy.linalg.eig()`](jax.numpy.linalg.eig.html#jax.numpy.linalg.eig "jax.numpy.linalg.eig"): general eigenvalue decomposition.

- [`jax.numpy.linalg.eigvalsh()`](jax.numpy.linalg.eigvalsh.html#jax.numpy.linalg.eigvalsh "jax.numpy.linalg.eigvalsh"): compute eigenvalues only.

- [`jax.scipy.linalg.eigh()`](jax.scipy.linalg.eigh.html#jax.scipy.linalg.eigh "jax.scipy.linalg.eigh"): SciPy API for Hermitian eigendecomposition.

- [`jax.lax.linalg.eigh()`](jax.lax.linalg.eigh.html#jax.lax.linalg.eigh "jax.lax.linalg.eigh"): XLA API for Hermitian eigendecomposition.

Examples

    >>> a = jnp.array([[1, -2j],
    ...                [2j, 1]])
    >>> w, v = jnp.linalg.eigh(a)
    >>> w
    Array([-1.,  3.], dtype=float32)
    >>> with jnp.printoptions(precision=3):
    ...   v
    Array([[-0.707+0.j   , -0.707+0.j   ],
           [ 0.   +0.707j,  0.   -0.707j]], dtype=complex64)

[](jax.numpy.linalg.eig.html "previous page")

previous

jax.numpy.linalg.eig

[](jax.numpy.linalg.eigvals.html "next page")

next

jax.numpy.linalg.eigvals

Contents

- [`eigh()`](#jax.numpy.linalg.eigh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
