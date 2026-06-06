- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.qr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.qr.rst "Download source file")
-  .pdf

# jax.scipy.linalg.qr

## Contents

- [`qr()`](#jax.scipy.linalg.qr)

# jax.scipy.linalg.qr[\#](#jax-scipy-linalg-qr "Link to this heading")

jax.scipy.linalg.qr(*a: ArrayLike*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *lwork: Any = None*, *\**, *mode: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['full', 'economic'\]*, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L958-L1040)[\#](#jax.scipy.linalg.qr "Link to this definition")\
jax.scipy.linalg.qr(*a: ArrayLike*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *lwork: Any = None*, *\**, *mode: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['full', 'economic'\]*, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\] = True*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.linalg.qr(*a: ArrayLike*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *lwork: Any = None*, *\**, *mode: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['full', 'economic'\]*, *pivoting: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\] \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.linalg.qr(*a: ArrayLike*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *lwork: Any = None*, *\**, *mode: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['r'\]*, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.linalg.qr(*a: ArrayLike*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *lwork: Any = None*, *\**, *mode: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['r'\]*, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\] = True*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.linalg.qr(*a: ArrayLike*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *lwork: Any = None*, *\**, *mode: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\['r'\]*, *pivoting: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\] \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.linalg.qr(*a: ArrayLike*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *lwork: Any = None*, *mode: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'full'*, *pivoting: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\] \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\] \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Compute the QR decomposition of an array

JAX implementation of [`scipy.linalg.qr()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.qr.html#scipy.linalg.qr "(in SciPy v1.19.0.dev)").

The QR decomposition of a matrix A is given by

\\A = QR\\

Where Q is a unitary matrix (i.e. \\Q^HQ=I\\) and R is an upper-triangular matrix.

Parameters:  
- **a** – array of shape (…, M, N)

- **mode** –

  Computational mode. Supported values are:

  - `"full"` (default): return Q of shape `(M,`` ``M)` and R of shape `(M,`` ``N)`.

  - `"r"`: return only R

  - `"economic"`: return Q of shape `(M,`` ``K)` and R of shape `(K,`` ``N)`, where K = min(M, N).

- **pivoting** – Allows the QR decomposition to be rank-revealing. If `True`, compute the column-pivoted decomposition `A[:,`` ``P]`` ``=`` ``Q`` ``@`` ``R`, where `P` is chosen such that the diagonal of `R` is non-increasing.

- **overwrite_a** – unused in JAX

- **lwork** – unused in JAX

- **check_finite** – unused in JAX

Returns:  
A tuple `(Q,`` ``R)` or `(Q,`` ``R,`` ``P)`, if `mode` is not `"r"` and `pivoting` is respectively `False` or `True`, otherwise an array `R` or tuple `(R,`` ``P)` if mode is `"r"`, and `pivoting` is respectively `False` or `True`, where:

- `Q` is an orthogonal matrix of shape `(...,`` ``M,`` ``M)` (if `mode` is `"full"`) or `(...,`` ``M,`` ``K)` (if `mode` is `"economic"`),

- `R` is an upper-triangular matrix of shape `(...,`` ``M,`` ``N)` (if `mode` is `"r"` or `"full"`) or `(...,`` ``K,`` ``N)` (if `mode` is `"economic"`),

- `P` is an index vector of shape `(...,`` ``N)`.

with `K`` ``=`` ``min(M,`` ``N)`.

Notes

- At present, pivoting is only implemented on the CPU and GPU backends. For further details about the GPU implementation, see the documentation for [`jax.lax.linalg.qr()`](jax.lax.linalg.qr.html#jax.lax.linalg.qr "jax.lax.linalg.qr").

See also

- [`jax.numpy.linalg.qr()`](jax.numpy.linalg.qr.html#jax.numpy.linalg.qr "jax.numpy.linalg.qr"): NumPy-style QR decomposition API

- [`jax.lax.linalg.qr()`](jax.lax.linalg.qr.html#jax.lax.linalg.qr "jax.lax.linalg.qr"): XLA-style QR decomposition API

Examples

Compute the QR decomposition of a matrix:

    >>> a = jnp.array([[1., 2., 3., 4.],
    ...                [5., 4., 2., 1.],
    ...                [6., 3., 1., 5.]])
    >>> Q, R = jax.scipy.linalg.qr(a)
    >>> Q  
    Array([[-0.12700021, -0.7581426 , -0.6396022 ],
           [-0.63500065, -0.43322435,  0.63960224],
           [-0.7620008 ,  0.48737738, -0.42640156]], dtype=float32)
    >>> R  
    Array([[-7.8740077, -5.080005 , -2.4130025, -4.953006 ],
           [ 0.       , -1.7870499, -2.6534991, -1.028908 ],
           [ 0.       ,  0.       , -1.0660033, -4.050814 ]], dtype=float32)

Check that `Q` is orthonormal:

    >>> jnp.allclose(Q.T @ Q, jnp.eye(3), atol=1E-5)
    Array(True, dtype=bool)

Reconstruct the input:

    >>> jnp.allclose(Q @ R, a)
    Array(True, dtype=bool)

[](jax.scipy.linalg.polar.html "previous page")

previous

jax.scipy.linalg.polar

[](jax.scipy.linalg.qr_multiply.html "next page")

next

jax.scipy.linalg.qr_multiply

Contents

- [`qr()`](#jax.scipy.linalg.qr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
