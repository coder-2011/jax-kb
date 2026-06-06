- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.schur

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.schur.rst "Download source file")
-  .pdf

# jax.scipy.linalg.schur

## Contents

- [`schur()`](#jax.scipy.linalg.schur)

# jax.scipy.linalg.schur[\#](#jax-scipy-linalg-schur "Link to this heading")

jax.scipy.linalg.schur(*a*, *output='real'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L512-L578)[\#](#jax.scipy.linalg.schur "Link to this definition")  
Compute the Schur decomposition

Only implemented on CPU.

JAX implementation of [`scipy.linalg.schur()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.schur.html#scipy.linalg.schur "(in SciPy v1.19.0.dev)").

The Schur form T of a matrix A satisfies:

\\A = Z T Z^H\\

where Z is unitary, and T is upper-triangular for the complex-valued Schur decomposition (i.e. `output="complex"`) and is quasi-upper-triangular for the real-valued Schur decomposition (i.e. `output="real"`). In the quasi-triangular case, the diagonal may include 2x2 blocks associated with complex-valued eigenvalue pairs of A.

Parameters:  
- **a** (*ArrayLike*) – input array of shape `(...,`` ``N,`` ``N)`

- **output** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – Specify whether to compute the `"real"` (default) or `"complex"` Schur decomposition.

Returns:  
A tuple of arrays `(T,`` ``Z)`

- `T` is a shape `(...,`` ``N,`` ``N)` array containing the upper-triangular Schur form of the input.

- `Z` is a shape `(...,`` ``N,`` ``N)` array containing the unitary Schur transformation matrix.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

See also

- [`jax.scipy.linalg.rsf2csf()`](jax.scipy.linalg.rsf2csf.html#jax.scipy.linalg.rsf2csf "jax.scipy.linalg.rsf2csf"): convert real Schur form to complex Schur form.

- [`jax.lax.linalg.schur()`](jax.lax.linalg.schur.html#jax.lax.linalg.schur "jax.lax.linalg.schur"): XLA-style API for Schur decomposition.

Examples

A Schur decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [1., 4., 2.],
    ...                [3., 2., 1.]])
    >>> T, Z = jax.scipy.linalg.schur(a)

The Schur form `T` is quasi-upper-triangular in general, but is truly upper-triangular in this case because the input matrix is symmetric:

    >>> T  
    Array([[-2.0000005 ,  0.5066295 , -0.43360388],
           [ 0.        ,  1.5505103 ,  0.74519426],
           [ 0.        ,  0.        ,  6.449491  ]], dtype=float32)

The transformation matrix `Z` is unitary:

    >>> jnp.allclose(Z.T @ Z, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

The input can be reconstructed from the outputs:

    >>> jnp.allclose(Z @ T @ Z.T, a)
    Array(True, dtype=bool)

[](jax.scipy.linalg.rsf2csf.html "previous page")

previous

jax.scipy.linalg.rsf2csf

[](jax.scipy.linalg.solve.html "next page")

next

jax.scipy.linalg.solve

Contents

- [`schur()`](#jax.scipy.linalg.schur)

By The JAX authors

© Copyright 2024, The JAX Authors.\
