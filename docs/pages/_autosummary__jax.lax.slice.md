- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.slice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.slice.rst "Download source file")
-  .pdf

# jax.lax.slice

## Contents

- [`slice()`](#jax.lax.slice)

# jax.lax.slice[\#](#jax-lax-slice "Link to this heading")

jax.lax.slice(*operand*, *start_indices*, *limit_indices*, *strides=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L62-L116)[\#](#jax.lax.slice "Link to this definition")  
Wraps XLA’s [Slice](https://www.openxla.org/xla/operation_semantics#slice) operator.

Parameters:  
- **operand** (*ArrayLike*) – an array to slice

- **start_indices** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of `operand.ndim` start indices.

- **limit_indices** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – a sequence of `operand.ndim` limit indices.

- **strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – an optional sequence of `operand.ndim` strides.

Returns:  
The sliced array

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Here are some examples of simple two-dimensional slices:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> lax.slice(x, (1, 0), (3, 2))
    Array([[4, 5],
           [8, 9]], dtype=int32)

    >>> lax.slice(x, (0, 0), (3, 4), (1, 2))
    Array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]], dtype=int32)

These two examples are equivalent to the following Python slicing syntax:

    >>> x[1:3, 0:2]
    Array([[4, 5],
           [8, 9]], dtype=int32)

    >>> x[0:3, 0:4:2]
    Array([[ 0,  2],
           [ 4,  6],
           [ 8, 10]], dtype=int32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- [`jax.lax.slice_in_dim()`](jax.lax.slice_in_dim.html#jax.lax.slice_in_dim "jax.lax.slice_in_dim")

- [`jax.lax.index_in_dim()`](jax.lax.index_in_dim.html#jax.lax.index_in_dim "jax.lax.index_in_dim")

- [`jax.lax.dynamic_slice()`](jax.lax.dynamic_slice.html#jax.lax.dynamic_slice "jax.lax.dynamic_slice")

[](jax.lax.sinh.html "previous page")

previous

jax.lax.sinh

[](jax.lax.slice_in_dim.html "next page")

next

jax.lax.slice_in_dim

Contents

- [`slice()`](#jax.lax.slice)

By The JAX authors

© Copyright 2024, The JAX Authors.\
