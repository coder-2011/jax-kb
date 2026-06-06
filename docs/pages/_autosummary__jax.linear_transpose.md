- [](../index.html)
- [API Reference](../jax.html)
- jax.linear_transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.linear_transpose.rst "Download source file")
-  .pdf

# jax.linear_transpose

## Contents

- [`linear_transpose()`](#jax.linear_transpose)

# jax.linear_transpose[\#](#jax-linear-transpose "Link to this heading")

jax.linear_transpose(*fun*, *\*primals*, *reduce_axes=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L1780-L1857)[\#](#jax.linear_transpose "Link to this definition")  
Transpose a function that is promised to be linear.

For linear functions, this transformation is equivalent to [`vjp()`](jax.vjp.html#jax.vjp "jax.vjp"), but avoids the overhead of computing the forward pass.

The outputs of the transposed function will always have the exact same dtypes as `primals`, even if some values are truncated (e.g., from complex to float, or from float64 to float32). To avoid truncation, use dtypes in `primals` that match the full range of desired outputs from the transposed function. Integer dtypes are not supported.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – the linear function to be transposed.

- **\*primals** – a positional argument tuple of arrays, scalars, or (nested) standard Python containers (tuples, lists, dicts, namedtuples, i.e., pytrees) of those types used for evaluating the shape/dtype of `fun(*primals)`. These arguments may be real scalars/ndarrays, but that is not required: only the `shape` and `dtype` attributes are accessed. See below for an example. (Note that the duck-typed objects cannot be namedtuples because those are treated as standard Python containers.)

Returns:  
A callable that calculates the transpose of `fun`. Valid input into this function must have the same shape/dtypes/structure as the result of `fun(*primals)`. Output will be a tuple, with the same shape/dtypes/structure as `primals`.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")

    >>> import jax
    >>>
    >>> f = lambda x, y: 0.5 * x - 0.5 * y
    >>> scalar = jax.ShapeDtypeStruct(shape=(), dtype=np.dtype(np.float32))
    >>> f_transpose = jax.linear_transpose(f, scalar, scalar)
    >>> f_transpose(1.0)
    (Array(0.5, dtype=float32), Array(-0.5, dtype=float32))

[](jax.linearize.html "previous page")

previous

jax.linearize

[](jax.vjp.html "next page")

next

jax.vjp

Contents

- [`linear_transpose()`](#jax.linear_transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
