- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.trace

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.trace.rst "Download source file")
-  .pdf

# jax.numpy.linalg.trace

## Contents

- [`trace()`](#jax.numpy.linalg.trace)

# jax.numpy.linalg.trace[\#](#jax-numpy-linalg-trace "Link to this heading")

jax.numpy.linalg.trace(*x*, */*, *\**, *offset=0*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L2263-L2306)[\#](#jax.numpy.linalg.trace "Link to this definition")  
Compute the trace of a matrix.

JAX implementation of [`numpy.linalg.trace()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.trace.html#numpy.linalg.trace "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – array of shape `(...,`` ``M,`` ``N)` and whose innermost two dimensions form MxN matrices for which to take the trace.

- **offset** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – positive or negative offset from the main diagonal (default: 0).

- **dtype** (*DTypeLike* *\|* *None*) – data type of the returned array (default: `None`). If `None`, then output dtype will match the dtype of `x`, promoted to default precision in the case of integer types.

Returns:  
array of batched traces with shape `x.shape[:-2]`

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.trace()`](jax.numpy.trace.html#jax.numpy.trace "jax.numpy.trace"): similar API in the `jax.numpy` namespace.

Examples

Trace of a single matrix:

    >>> x = jnp.array([[1,  2,  3,  4],
    ...                [5,  6,  7,  8],
    ...                [9, 10, 11, 12]])
    >>> jnp.linalg.trace(x)
    Array(18, dtype=int32)
    >>> jnp.linalg.trace(x, offset=1)
    Array(21, dtype=int32)
    >>> jnp.linalg.trace(x, offset=-1, dtype="float32")
    Array(15., dtype=float32)

Batched traces:

    >>> x = jnp.arange(24).reshape(2, 3, 4)
    >>> jnp.linalg.trace(x)
    Array([15, 51], dtype=int32)

[](jax.numpy.linalg.tensorsolve.html "previous page")

previous

jax.numpy.linalg.tensorsolve

[](jax.numpy.linalg.vector_norm.html "next page")

next

jax.numpy.linalg.vector_norm

Contents

- [`trace()`](#jax.numpy.linalg.trace)

By The JAX authors

© Copyright 2024, The JAX Authors.\
