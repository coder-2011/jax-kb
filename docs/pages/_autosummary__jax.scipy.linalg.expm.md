- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.expm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.expm.rst "Download source file")
-  .pdf

# jax.scipy.linalg.expm

## Contents

- [`expm()`](#jax.scipy.linalg.expm)

# jax.scipy.linalg.expm[\#](#jax-scipy-linalg-expm "Link to this heading")

jax.scipy.linalg.expm(*A*, *\**, *upper_triangular=False*, *max_squarings=16*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L1371-L1445)[\#](#jax.scipy.linalg.expm "Link to this definition")  
Compute the matrix exponential

JAX implementation of [`scipy.linalg.expm()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.expm.html#scipy.linalg.expm "(in SciPy v1.19.0.dev)").

Parameters:  
- **A** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)`

- **upper_triangular** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then assume that `A` is upper-triangular. Default=False.

- **max_squarings** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The number of squarings in the scaling-and-squaring approximation method (default: 16).

Returns:  
An array of shape `(...,`` ``N,`` ``N)` containing the matrix exponent of `A`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

This uses the scaling-and-squaring approximation method, with computational complexity controlled by the optional `max_squarings` argument. Theoretically, the number of required squarings is `max(0,`` ``ceil(log2(norm(A)))`` ``-`` ``c)` where `norm(A)` is the L1 norm and `c=2.42` for float64/complex128, or `c=1.97` for float32/complex64.

See also

[`jax.scipy.linalg.expm_frechet()`](jax.scipy.linalg.expm_frechet.html#jax.scipy.linalg.expm_frechet "jax.scipy.linalg.expm_frechet")

Examples

`expm` is the matrix exponential, and has similar properties to the more familiar scalar exponential. For scalars `a` and `b`, \\e^{a + b} = e^a e^b\\. However, for matrices, this property only holds when `A` and `B` commute (`AB`` ``=`` ``BA`). In this case, `expm(A+B)`` ``=`` ``expm(A)`` ``@`` ``expm(B)`

    >>> A = jnp.array([[2, 0],
    ...                [0, 1]])
    >>> B = jnp.array([[3, 0],
    ...                [0, 4]])
    >>> jnp.allclose(jax.scipy.linalg.expm(A+B),
    ...              jax.scipy.linalg.expm(A) @ jax.scipy.linalg.expm(B),
    ...              rtol=0.0001)
    Array(True, dtype=bool)

If a matrix `X` is invertible, then `expm(X`` ``@`` ``A`` ``@`` ``inv(X))`` ``=`` ``X`` ``@`` ``expm(A)`` ``@`` ``inv(X)`

    >>> X = jnp.array([[3, 1],
    ...                [2, 5]])
    >>> X_inv = jax.scipy.linalg.inv(X)
    >>> jnp.allclose(jax.scipy.linalg.expm(X @ A @ X_inv),
    ...              X @ jax.scipy.linalg.expm(A) @ X_inv)
    Array(True, dtype=bool)

[](jax.scipy.linalg.eigh_tridiagonal.html "previous page")

previous

jax.scipy.linalg.eigh_tridiagonal

[](jax.scipy.linalg.expm_frechet.html "next page")

next

jax.scipy.linalg.expm_frechet

Contents

- [`expm()`](#jax.scipy.linalg.expm)

By The JAX authors

© Copyright 2024, The JAX Authors.\
