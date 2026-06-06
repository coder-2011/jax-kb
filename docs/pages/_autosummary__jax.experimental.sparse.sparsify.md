- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.sparsify

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.sparsify.rst "Download source file")
-  .pdf

# jax.experimental.sparse.sparsify

## Contents

- [`sparsify()`](#jax.experimental.sparse.sparsify)

# jax.experimental.sparse.sparsify[\#](#jax-experimental-sparse-sparsify "Link to this heading")

jax.experimental.sparse.sparsify(*f*, *use_tracer=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/transform.py#L484-L509)[\#](#jax.experimental.sparse.sparsify "Link to this definition")  
Experimental sparsification transform.

Examples

Decorate JAX functions to make them compatible with [`jax.experimental.sparse.BCOO`](jax.experimental.sparse.BCOO.html#jax.experimental.sparse.BCOO "jax.experimental.sparse.BCOO") matrices:

    >>> from jax.experimental import sparse

    >>> @sparse.sparsify
    ... def f(M, v):
    ...   return 2 * M.T @ v

    >>> M = sparse.BCOO.fromdense(jnp.arange(12).reshape(3, 4))

    >>> v = jnp.array([3, 4, 2])

    >>> f(M, v)
    Array([ 64,  82, 100, 118], dtype=int32)

[](../jax.experimental.sparse.html "previous page")

previous

`jax.experimental.sparse` module

[](jax.experimental.sparse.grad.html "next page")

next

jax.experimental.sparse.grad

Contents

- [`sparsify()`](#jax.experimental.sparse.sparsify)

By The JAX authors

© Copyright 2024, The JAX Authors.\
