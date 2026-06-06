- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.qr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.qr.rst "Download source file")
-  .pdf

# jax.lax.linalg.qr

## Contents

- [`qr()`](#jax.lax.linalg.qr)

# jax.lax.linalg.qr[\#](#jax-lax-linalg-qr "Link to this heading")

jax.lax.linalg.qr(*x: ArrayLike*, *\**, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *use_magma: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L395-L447)[\#](#jax.lax.linalg.qr "Link to this definition")\
jax.lax.linalg.qr(*x: ArrayLike*, *\**, *pivoting: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *use_magma: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.lax.linalg.qr(*x: ArrayLike*, *\**, *pivoting: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *use_magma: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\] \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
QR decomposition.

Computes the QR decomposition

\\A = Q \\ R\\

of matrices \\A\\, such that \\Q\\ is a unitary (orthogonal) matrix, and \\R\\ is an upper-triangular matrix.

Parameters:  
- **x** – A batch of matrices with shape `[...,`` ``m,`` ``n]`.

- **pivoting** – Allows the QR decomposition to be rank-revealing. If `True`, compute the column pivoted decomposition `A[:,`` ``P]`` ``=`` ``Q`` ``@`` ``R`, where `P` is chosen such that the diagonal of `R` is non-increasing. Currently supported on CPU and GPU backends only.

- **full_matrices** – Determines if full or reduced matrices are returned; see below.

- **use_magma** – Locally override the `jax_use_magma` flag. If `True`, the pivoted qr factorization is computed using MAGMA. If `False`, the computation is done using LAPACK on the host CPU. If `None` (default), the behavior is controlled by the `jax_use_magma` flag. This argument is only used on GPU.

Returns:  
A pair of arrays `(q,`` ``r)`, if `pivoting=False`, otherwise `(q,`` ``r,`` ``p)`.

Array `q` is a unitary (orthogonal) matrix, with shape `[...,`` ``m,`` ``m]` if `full_matrices=True`, or `[...,`` ``m,`` ``min(m,`` ``n)]` if `full_matrices=False`.

Array `r` is an upper-triangular matrix with shape `[...,`` ``m,`` ``n]` if `full_matrices=True`, or `[...,`` ``min(m,`` ``n),`` ``n]` if `full_matrices=False`.

Array `p` is an index vector with shape \[…, n\]

Notes

- [MAGMA](https://icl.utk.edu/magma/) support is experimental - see [`jax.lax.linalg.eig()`](jax.lax.linalg.eig.html#jax.lax.linalg.eig "jax.lax.linalg.eig") for further assumptions and limitations.

- If `jax_use_magma` is set to `"auto"`, the MAGMA implementation will be used if the library can be found, and the input matrix is sufficiently large (has at least 2048 columns).

[](jax.lax.linalg.qdwh.html "previous page")

previous

jax.lax.linalg.qdwh

[](jax.lax.linalg.schur.html "next page")

next

jax.lax.linalg.schur

Contents

- [`qr()`](#jax.lax.linalg.qr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
