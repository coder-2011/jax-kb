- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.eigvalsh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.eigvalsh.rst "Download source file")
-  .pdf

# jax.numpy.linalg.eigvalsh

## Contents

- [`eigvalsh()`](#jax.numpy.linalg.eigvalsh)

# jax.numpy.linalg.eigvalsh[\#](#jax-numpy-linalg-eigvalsh "Link to this heading")

jax.numpy.linalg.eigvalsh(*a*, *UPLO='L'*, *\**, *symmetrize_input=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L944-L983)[\#](#jax.numpy.linalg.eigvalsh "Link to this definition")  
Compute the eigenvalues of a Hermitian matrix.

JAX implementation of [`numpy.linalg.eigvalsh()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``M)`, containing the Hermitian (if complex) or symmetric (if real) matrix.

- **UPLO** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – specifies whether the calculation is done with the lower triangular part of `a` (`'L'`, default) or the upper triangular part (`'U'`).

- **symmetrize_input** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True (default) then input is symmetrized, which leads to better behavior under automatic differentiation. Note that when this is set to True, both the upper and lower triangles of the input will be used in computing the decomposition.

Returns:  
An array of shape `(...,`` ``M)` containing the eigenvalues, sorted in ascending order.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.eig()`](jax.numpy.linalg.eig.html#jax.numpy.linalg.eig "jax.numpy.linalg.eig"): general eigenvalue decomposition.

- [`jax.numpy.linalg.eigh()`](jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh "jax.numpy.linalg.eigh"): computes eigenvalues and eigenvectors of a Hermitian matrix.

Examples

    >>> a = jnp.array([[1, -2j],
    ...                [2j, 1]])
    >>> w = jnp.linalg.eigvalsh(a)
    >>> w
    Array([-1.,  3.], dtype=float32)

[](jax.numpy.linalg.eigvals.html "previous page")

previous

jax.numpy.linalg.eigvals

[](jax.numpy.linalg.inv.html "next page")

next

jax.numpy.linalg.inv

Contents

- [`eigvalsh()`](#jax.numpy.linalg.eigvalsh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
