- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.rsf2csf

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.rsf2csf.rst "Download source file")
-  .pdf

# jax.scipy.linalg.rsf2csf

## Contents

- [`rsf2csf()`](#jax.scipy.linalg.rsf2csf)

# jax.scipy.linalg.rsf2csf[\#](#jax-scipy-linalg-rsf2csf "Link to this heading")

jax.scipy.linalg.rsf2csf(*T*, *Z*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2274-L2338)[\#](#jax.scipy.linalg.rsf2csf "Link to this definition")  
Convert real Schur form to complex Schur form.

JAX implementation of [`scipy.linalg.rsf2csf()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.rsf2csf.html#scipy.linalg.rsf2csf "(in SciPy v1.19.0.dev)").

Parameters:  
- **T** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)` containing the real Schur form of the input.

- **Z** (*ArrayLike*) – array of shape `(...,`` ``N,`` ``N)` containing the corresponding Schur transformation matrix. Batch dimensions are broadcast with those of `T`.

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

Returns:  
A tuple of arrays `(T,`` ``Z)` of the same shape as the inputs, containing the Complex Schur form and the associated Schur transformation matrix.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

[`jax.scipy.linalg.schur()`](jax.scipy.linalg.schur.html#jax.scipy.linalg.schur "jax.scipy.linalg.schur"): Schur decomposition

Examples

    >>> A = jnp.array([[0., 3., 3.],
    ...                [0., 1., 2.],
    ...                [2., 0., 1.]])
    >>> Tr, Zr = jax.scipy.linalg.schur(A)
    >>> Tc, Zc = jax.scipy.linalg.rsf2csf(Tr, Zr)

Both the real and complex form can be used to reconstruct the input matrix to float32 precision:

    >>> jnp.allclose(Zr @ Tr @ Zr.T, A, atol=1E-5)
    Array(True, dtype=bool)
    >>> jnp.allclose(Zc @ Tc @ Zc.conj().T, A, atol=1E-5)
    Array(True, dtype=bool)

The real-valued Schur form is only quasi-upper-triangular, as we can see in this case:

    >>> with jax.numpy.printoptions(precision=2, suppress=True):
    ...   print(Tr)
    [[ 3.76 -2.17  1.38]
     [ 0.   -0.88 -0.35]
     [ 0.    2.37 -0.88]]

By contrast, the complex form is truly upper-triangular:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(Tc)
    [[ 3.76+0.j    1.29-0.78j  2.02-0.5j ]
     [ 0.  +0.j   -0.88+0.91j -2.02+0.j  ]
     [ 0.  +0.j    0.  +0.j   -0.88-0.91j]]

[](jax.scipy.linalg.qr_multiply.html "previous page")

previous

jax.scipy.linalg.qr_multiply

[](jax.scipy.linalg.schur.html "next page")

next

jax.scipy.linalg.schur

Contents

- [`rsf2csf()`](#jax.scipy.linalg.rsf2csf)

By The JAX authors

© Copyright 2024, The JAX Authors.\
