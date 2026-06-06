- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.lu

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.lu.rst "Download source file")
-  .pdf

# jax.scipy.linalg.lu

## Contents

- [`lu()`](#jax.scipy.linalg.lu)

# jax.scipy.linalg.lu[\#](#jax-scipy-linalg-lu "Link to this heading")

jax.scipy.linalg.lu(*a: ArrayLike*, *permute_l: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\] = False*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L800-L870)[\#](#jax.scipy.linalg.lu "Link to this definition")\
jax.scipy.linalg.lu(*a: ArrayLike*, *permute_l: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]\
jax.scipy.linalg.lu(*a: ArrayLike*, *permute_l: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *overwrite_a: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *check_finite: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\] \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Compute the LU decomposition

JAX implementation of [`scipy.linalg.lu()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.lu.html#scipy.linalg.lu "(in SciPy v1.19.0.dev)").

The LU decomposition of a matrix A is:

\\A = P L U\\

where P is a permutation matrix, L is lower-triangular and U is upper-triangular.

Parameters:  
- **a** – array of shape `(...,`` ``M,`` ``N)` to decompose.

- **permute_l** – if True, then permute `L` and return `(P`` ``@`` ``L,`` ``U)` (default: False)

- **overwrite_a** – not used by JAX

- **check_finite** – not used by JAX

Returns:  
- `P` is a permutation matrix of shape `(...,`` ``M,`` ``M)`

- `L` is a lower-triangular matrix of shape `(...`` ``M,`` ``K)`

- `U` is an upper-triangular matrix of shape `(...,`` ``K,`` ``N)`

with `K`` ``=`` ``min(M,`` ``N)`

Return type:  
A tuple of arrays `(P`` ``@`` ``L,`` ``U)` if `permute_l` is True, else `(P,`` ``L,`` ``U)`

See also

- `jax.numpy.linalg.lu()`: NumPy-style API for LU decomposition.

- [`jax.lax.linalg.lu()`](jax.lax.linalg.lu.html#jax.lax.linalg.lu "jax.lax.linalg.lu"): XLA-style API for LU decomposition.

- [`jax.scipy.linalg.lu_solve()`](jax.scipy.linalg.lu_solve.html#jax.scipy.linalg.lu_solve "jax.scipy.linalg.lu_solve"): LU-based linear solver.

Examples

An LU decomposition of a 3x3 matrix:

    >>> a = jnp.array([[1., 2., 3.],
    ...                [5., 4., 2.],
    ...                [3., 2., 1.]])
    >>> P, L, U = jax.scipy.linalg.lu(a)

`P` is a permutation matrix: i.e. each row and column has a single `1`:

    >>> P
    Array([[0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.]], dtype=float32)

`L` and `U` are lower-triangular and upper-triangular matrices:

    >>> with jnp.printoptions(precision=3):
    ...   print(L)
    ...   print(U)
    [[ 1.     0.     0.   ]
     [ 0.2    1.     0.   ]
     [ 0.6   -0.333  1.   ]]
    [[5.    4.    2.   ]
     [0.    1.2   2.6  ]
     [0.    0.    0.667]]

The original matrix can be reconstructed by multiplying the three together:

    >>> a_reconstructed = P @ L @ U
    >>> jnp.allclose(a, a_reconstructed)
    Array(True, dtype=bool)

[](jax.scipy.linalg.leslie.html "previous page")

previous

jax.scipy.linalg.leslie

[](jax.scipy.linalg.lu_factor.html "next page")

next

jax.scipy.linalg.lu_factor

Contents

- [`lu()`](#jax.scipy.linalg.lu)

By The JAX authors

© Copyright 2024, The JAX Authors.\
