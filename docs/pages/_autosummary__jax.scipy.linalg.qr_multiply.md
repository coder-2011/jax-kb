- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.qr_multiply

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.qr_multiply.rst "Download source file")
-  .pdf

# jax.scipy.linalg.qr_multiply

## Contents

- [`qr_multiply()`](#jax.scipy.linalg.qr_multiply)

# jax.scipy.linalg.qr_multiply[\#](#jax-scipy-linalg-qr-multiply "Link to this heading")

jax.scipy.linalg.qr_multiply(*a: ArrayLike*, *c: ArrayLike*, *mode: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'right'*, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *conjugate: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_c: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L1061-L1180)[\#](#jax.scipy.linalg.qr_multiply "Link to this definition")\
jax.scipy.linalg.qr_multiply(*a: ArrayLike*, *c: ArrayLike*, *mode: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'right'*, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\] = True*, *conjugate: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_c: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.linalg.qr_multiply(*a: ArrayLike*, *c: ArrayLike*, *mode: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") = 'right'*, *pivoting: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *conjugate: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_c: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\] \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Calculate the QR decomposition and multiply Q with a matrix.

JAX implementation of [`scipy.linalg.qr_multiply()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.qr_multiply.html#scipy.linalg.qr_multiply "(in SciPy v1.19.0.dev)").

Parameters:  
- **a** – array of shape `(...,`` ``M,`` ``N)`. Matrix to be decomposed.

- **c** – array to be multiplied by Q. For `mode='left'`, `c` has shape `(...,`` ``K,`` ``P)` where `K`` ``=`` ``min(M,`` ``N)`. For `mode='right'`, `c` has shape `(...,`` ``P,`` ``M)`. 1-D arrays are supported: for `mode='left'`, treated as a length-`K` column vector; for `mode='right'`, treated as a length-`M` row vector. The result is raveled back to 1-D in either case.

- **mode** –

  `'right'` (default) or `'left'`.

  - `'left'`: compute `Q`` ``@`` ``c` (or `conj(Q)`` ``@`` ``c` if `conjugate=True`) and return `(Q`` ``@`` ``c,`` ``R)` with result shape `(...,`` ``M,`` ``P)`

  - `'right'`: compute `c`` ``@`` ``Q` (or `c`` ``@`` ``conj(Q)` if `conjugate=True`) and return `(c`` ``@`` ``Q,`` ``R)` with result shape `(...,`` ``P,`` ``K)` where `K`` ``=`` ``min(M,`` ``N)`

- **pivoting** – Allows the QR decomposition to be rank-revealing. If `True`, compute the column-pivoted QR decomposition and return permutation indices as a third element.

- **conjugate** – If `True`, use `conj(Q)` (element-wise complex conjugate) instead of `Q`. For real arrays this has no effect.

- **overwrite_a** – unused in JAX

- **overwrite_c** – unused in JAX

Returns:  
`(result,`` ``R)`

If `pivoting` is `True`: `(result,`` ``R,`` ``P)`

Return type:  
If `pivoting` is `False`

See also

- [`jax.scipy.linalg.qr()`](jax.scipy.linalg.qr.html#jax.scipy.linalg.qr "jax.scipy.linalg.qr"): SciPy-style QR decomposition API

- [`jax.lax.linalg.ormqr()`](jax.lax.linalg.ormqr.html#jax.lax.linalg.ormqr "jax.lax.linalg.ormqr"): XLA-style Q-multiply primitive

Examples

Use [`qr_multiply()`](#jax.scipy.linalg.qr_multiply "jax.scipy.linalg.qr_multiply") to efficiently solve a least-squares problem. For an overdetermined system `A`` ``@`` ``x`` ``≈`` ``b`, pass `b` as a 1-D row via `mode='right'` to obtain `Q^T`` ``@`` ``b` and `R` in one step:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> A = jnp.array([[1., 1.], [1., 2.], [1., 3.], [1., 4.]])
    >>> b = jnp.array([2., 4., 5., 4.])
    >>> Qtb, R = jax.scipy.linalg.qr_multiply(A, b, mode='right')
    >>> x = jax.scipy.linalg.solve_triangular(R, Qtb)
    >>> jnp.allclose(A.T @ A @ x, A.T @ b)
    Array(True, dtype=bool)

[](jax.scipy.linalg.qr.html "previous page")

previous

jax.scipy.linalg.qr

[](jax.scipy.linalg.rsf2csf.html "next page")

next

jax.scipy.linalg.rsf2csf

Contents

- [`qr_multiply()`](#jax.scipy.linalg.qr_multiply)

By The JAX authors

© Copyright 2024, The JAX Authors.\
