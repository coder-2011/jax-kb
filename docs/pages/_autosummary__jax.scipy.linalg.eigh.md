- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.eigh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.eigh.rst "Download source file")
-  .pdf

# jax.scipy.linalg.eigh

## Contents

- [`eigh()`](#jax.scipy.linalg.eigh)

# jax.scipy.linalg.eigh[\#](#jax-scipy-linalg-eigh "Link to this heading")

jax.scipy.linalg.eigh(*a: ArrayLike*, *b: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *lower: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *eigvals_only: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_b: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *turbo: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *eigvals: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *type: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 1*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L438-L505)[\#](#jax.scipy.linalg.eigh "Link to this definition")\
jax.scipy.linalg.eigh(*a: ArrayLike*, *b: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *lower: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *\**, *eigvals_only: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_b: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *turbo: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *eigvals: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *type: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 1*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [Array](jax.Array.html#jax.Array "jax.Array")\
jax.scipy.linalg.eigh(*a: ArrayLike*, *b: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)")*, *lower: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*, *eigvals_only: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_b: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *turbo: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *eigvals: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *type: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 1*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [Array](jax.Array.html#jax.Array "jax.Array")\
jax.scipy.linalg.eigh(*a: ArrayLike*, *b: ArrayLike \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *lower: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *eigvals_only: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_b: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *turbo: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *eigvals: [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *type: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 1*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Compute eigenvalues and eigenvectors for a Hermitian matrix

JAX implementation of [`scipy.linalg.eigh()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.eigh.html#scipy.linalg.eigh "(in SciPy v1.19.0.dev)").

Only the standard eigenvalue problem is supported: `a`` ``@`` ``v`` ``=`` ``lambda`` ``*`` ``v`.  
The parameter b must be None; the generalized problem (`a`` ``@`` ``v`` ``=`` ``lambda`` ``*`` ``b`` ``@`` ``v`) is not implemented.

Parameters:  
- **a** – Hermitian input array of shape `(...,`` ``N,`` ``N)`

- **b** – Must be None. The generalized eigenvalue problem is not supported.

- **lower** – if True (default) access only the lower portion of the input matrix. Otherwise access only the upper portion.

- **eigvals_only** – If True, compute only the eigenvalues. If False (default) compute both eigenvalues and eigenvectors.

- **type** – Not used. Only type=1 is supported.

- **eigvals** – Not used. Only eigvals=None is supported.

- **overwrite_a** – unused by JAX.

- **overwrite_b** – unused by JAX.

- **turbo** – unused by JAX.

- **check_finite** – unused by JAX.

Returns:  
A tuple of arrays `(eigvals,`` ``eigvecs)` if `eigvals_only` is False, otherwise an array `eigvals`.

- `eigvals`: array of shape `(...,`` ``N)` containing the eigenvalues.

- `eigvecs`: array of shape `(...,`` ``N,`` ``N)` containing the eigenvectors.

Raises:  
[**NotImplementedError**](https://docs.python.org/3/library/exceptions.html#NotImplementedError "(in Python v3.14)") – If b is not None.

See also

- [`jax.numpy.linalg.eigh()`](jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh "jax.numpy.linalg.eigh"): NumPy-style eigh API.

- [`jax.lax.linalg.eigh()`](jax.lax.linalg.eigh.html#jax.lax.linalg.eigh "jax.lax.linalg.eigh"): XLA-style eigh API.

- [`jax.numpy.linalg.eig()`](jax.numpy.linalg.eig.html#jax.numpy.linalg.eig "jax.numpy.linalg.eig"): non-hermitian eigenvalue problem.

- [`jax.scipy.linalg.eigh_tridiagonal()`](jax.scipy.linalg.eigh_tridiagonal.html#jax.scipy.linalg.eigh_tridiagonal "jax.scipy.linalg.eigh_tridiagonal"): tri-diagonal eigenvalue problem.

Examples

Compute the standard eigenvalue decomposition of a simple 2x2 matrix:

    >>> a = jnp.array([[2., 1.],
    ...                [1., 2.]])
    >>> eigvals, eigvecs = jax.scipy.linalg.eigh(a)
    >>> eigvals
    Array([1., 3.], dtype=float32)
    >>> eigvecs
    Array([[-0.70710677,  0.70710677],
           [ 0.70710677,  0.70710677]], dtype=float32)

Eigenvectors are orthonormal:

    >>> jnp.allclose(eigvecs.T @ eigvecs, jnp.eye(2), atol=1E-5)
    Array(True, dtype=bool)

Solution satisfies the eigenvalue problem:

    >>> jnp.allclose(a @ eigvecs, eigvecs @ jnp.diag(eigvals))
    Array(True, dtype=bool)

[](jax.scipy.linalg.dft.html "previous page")

previous

jax.scipy.linalg.dft

[](jax.scipy.linalg.eigh_tridiagonal.html "next page")

next

jax.scipy.linalg.eigh_tridiagonal

Contents

- [`eigh()`](#jax.scipy.linalg.eigh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
