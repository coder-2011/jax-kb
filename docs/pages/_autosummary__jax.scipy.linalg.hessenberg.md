- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.hessenberg

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.hessenberg.rst "Download source file")
-  .pdf

# jax.scipy.linalg.hessenberg

## Contents

- [`hessenberg()`](#jax.scipy.linalg.hessenberg)

# jax.scipy.linalg.hessenberg[\#](#jax-scipy-linalg-hessenberg "Link to this heading")

jax.scipy.linalg.hessenberg(*a: ArrayLike*, *\**, *calc_q: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\]*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [Array](jax.Array.html#jax.Array "jax.Array")[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2348-L2415)[\#](#jax.scipy.linalg.hessenberg "Link to this definition")\
jax.scipy.linalg.hessenberg(*a: ArrayLike*, *\**, *calc_q: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Compute the Hessenberg form of the matrix

JAX implementation of [`scipy.linalg.hessenberg()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.hessenberg.html#scipy.linalg.hessenberg "(in SciPy v1.19.0.dev)").

The Hessenberg form H of a matrix A satisfies:

\\A = Q H Q^H\\

where Q is unitary and H is zero below the first subdiagonal.

Parameters:  
- **a** – array of shape `(...,`` ``N,`` ``N)`

- **calc_q** – if True, calculate the `Q` matrix (default: False)

- **overwrite_a** – unused by JAX

- **check_finite** – unused by JAX

Returns:  
A tuple of arrays `(H,`` ``Q)` if `calc_q` is True, else an array `H`

- `H` has shape `(...,`` ``N,`` ``N)` and is the Hessenberg form of `a`

- `Q` has shape `(...,`` ``N,`` ``N)` and is the associated unitary matrix

Examples

Computing the Hessenberg form of a 4x4 matrix

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [1., 4., 2., 3.],
    ...                [3., 2., 1., 4.],
    ...                [2., 3., 2., 2.]])
    >>> H, Q = jax.scipy.linalg.hessenberg(a, calc_q=True)
    >>> with jnp.printoptions(suppress=True, precision=3):
    ...   print(H)
    [[ 1.    -5.078  1.167  1.361]
     [-3.742  5.786 -3.613 -1.825]
     [ 0.    -2.992  2.493 -0.577]
     [ 0.     0.    -0.043 -1.279]]

Notice the zeros in the subdiagonal positions. The original matrix can be reconstructed using the `Q` vectors:

    >>> a_reconstructed = Q @ H @ Q.conj().T
    >>> jnp.allclose(a_reconstructed, a)
    Array(True, dtype=bool)

[](jax.scipy.linalg.helmert.html "previous page")

previous

jax.scipy.linalg.helmert

[](jax.scipy.linalg.hankel.html "next page")

next

jax.scipy.linalg.hankel

Contents

- [`hessenberg()`](#jax.scipy.linalg.hessenberg)

By The JAX authors

© Copyright 2024, The JAX Authors.\
