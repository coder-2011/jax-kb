- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.linalg.qdwh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.linalg.qdwh.rst "Download source file")
-  .pdf

# jax.lax.linalg.qdwh

## Contents

- [`qdwh()`](#jax.lax.linalg.qdwh)

# jax.lax.linalg.qdwh[\#](#jax-lax-linalg-qdwh "Link to this heading")

jax.lax.linalg.qdwh(*x*, *\**, *is_hermitian=False*, *max_iterations=None*, *eps=None*, *dynamic_shape=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tpu/linalg/qdwh.py#L237-L292)[\#](#jax.lax.linalg.qdwh "Link to this definition")  
QR-based dynamically weighted Halley iteration for polar decomposition.

Parameters:  
- **x** – A full-rank matrix, with shape M x N. The matrix may be padded up to that size from a smaller true shape (`dynamic_shape`).

- **is_hermitian** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – True if x is Hermitian. Default to False. This parameter is currently unused, but exists for backward compatibility.

- **eps** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *None*) – The final result will satisfy `|x_k`` ``-`` ``x_k-1|`` ``<`` ``|x_k|`` ``*`` ``(4*eps)**(1/3)` where x_k is the iterate.

- **max_iterations** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Iterations will terminate after this many steps even if the above is unsatisfied.

- **dynamic_shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – the unpadded shape as an `(m,`` ``n)` tuple; optional.

Returns:  
A four-tuple of (u, h, num_iters, is_converged) containing the polar decomposition of x = u \* h, the number of iterations to compute u, and is_converged, whose value is True when the convergence is achieved within the maximum number of iterations.

[](jax.lax.linalg.ormqr.html "previous page")

previous

jax.lax.linalg.ormqr

[](jax.lax.linalg.qr.html "next page")

next

jax.lax.linalg.qr

Contents

- [`qdwh()`](#jax.lax.linalg.qdwh)

By The JAX authors

© Copyright 2024, The JAX Authors.\
