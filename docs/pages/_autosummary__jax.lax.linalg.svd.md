- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.svd

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.svd.rst "Download source file")
-  .pdf

# jax.lax.linalg.svd

## Contents

- [`svd()`](#jax.lax.linalg.svd)

# jax.lax.linalg.svd[\#](#jax-lax-linalg-svd "Link to this heading")

jax.lax.linalg.svd(*x: ArrayLike*, *\**, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *compute_uv: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[True\]*, *subset_by_index: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *algorithm: [SvdAlgorithm](jax.lax.linalg.SvdAlgorithm.html#jax.lax.linalg.SvdAlgorithm "jax.lax.linalg.SvdAlgorithm") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\][\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/linalg.py#L531-L572)[\#](#jax.lax.linalg.svd "Link to this definition")\
jax.lax.linalg.svd(*x: ArrayLike*, *\**, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *compute_uv: [Literal](jax.extend.core.Literal.html#jax.extend.core.Literal "jax.extend.core.Literal")\[False\]*, *subset_by_index: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *algorithm: [SvdAlgorithm](jax.lax.linalg.SvdAlgorithm.html#jax.lax.linalg.SvdAlgorithm "jax.lax.linalg.SvdAlgorithm") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Array](jax.Array.html#jax.Array "jax.Array")\
jax.lax.linalg.svd(*x: ArrayLike*, *\**, *full_matrices: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *compute_uv: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *subset_by_index: [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *algorithm: [SvdAlgorithm](jax.lax.linalg.SvdAlgorithm.html#jax.lax.linalg.SvdAlgorithm "jax.lax.linalg.SvdAlgorithm") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]  
Singular value decomposition.

Computes the singular value decomposition of an input matrix.

Parameters:  
- **x** – A batch of matrices with shape `[...,`` ``m,`` ``n]`.

- **full_matrices** – Determines if full or reduced matrices are returned.

- **compute_uv** – If `True`, returns the left singular vectors, the singular values and the adjoint of the right singular vectors. Otherwise, only the singular values are returned.

- **subset_by_index** – If `None`, the entire matrix is returned. Otherwise, returns the singular values and vectors for the given range of indices.

- **algorithm** – The SVD algorithm to use. Must be `None` or a value from [`SvdAlgorithm`](jax.lax.linalg.SvdAlgorithm.html#jax.lax.linalg.SvdAlgorithm "jax.lax.linalg.SvdAlgorithm").

Returns:  
The singular values if `compute_uv` is `False`, otherwise returns a triple containing the left singular vectors, the singular values, and the adjoint of the right singular vectors.

[](jax.lax.linalg.schur.html "previous page")

previous

jax.lax.linalg.schur

[](jax.lax.linalg.SvdAlgorithm.html "next page")

next

jax.lax.linalg.SvdAlgorithm

Contents

- [`svd()`](#jax.lax.linalg.svd)

By The JAX authors

© Copyright 2024, The JAX Authors.\
