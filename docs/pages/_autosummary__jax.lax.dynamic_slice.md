- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.dynamic_slice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.dynamic_slice.rst "Download source file")
-  .pdf

# jax.lax.dynamic_slice

## Contents

- [`dynamic_slice()`](#jax.lax.dynamic_slice)

# jax.lax.dynamic_slice[\#](#jax-lax-dynamic-slice "Link to this heading")

jax.lax.dynamic_slice(*operand*, *start_indices*, *slice_sizes*, *\**, *allow_negative_indices=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L118-L179)[\#](#jax.lax.dynamic_slice "Link to this definition")  
Wraps XLA’s [DynamicSlice](https://www.openxla.org/xla/operation_semantics#dynamicslice) operator.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray*) – an array to slice.

- **start_indices** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray* *\|* *Sequence\[ArrayLike\]*) – a list of scalar indices, one per dimension. These values may be dynamic.

- **slice_sizes** (*Shape*) – the size of the slice. Must be a sequence of non-negative integers with length equal to ndim(operand). Inside a JIT compiled function, only static values are supported (all JAX arrays inside JIT must have statically known size).

- **allow_negative_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *Sequence\[*[*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*\]*) – a bool or sequence of bools, one per dimension; if a bool is passed, it applies to all dimensions. For each dimension, if true, negative indices are permitted and are are interpreted relative to the end of the array. If false, negative indices are treated as if they were out of bounds and the result is implementation defined, typically clamped to the first index.

Returns:  
An array containing the slice.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

Here is a simple two-dimensional dynamic slice:

    >>> x = jnp.arange(12).reshape(3, 4)
    >>> x
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

    >>> dynamic_slice(x, (1, 1), (2, 3))
    Array([[ 5,  6,  7],
           [ 9, 10, 11]], dtype=int32)

Note the potentially surprising behavior for the case where the requested slice overruns the bounds of the array; in this case the start index is adjusted to return a slice of the requested size:

    >>> dynamic_slice(x, (1, 1), (2, 4))
    Array([[ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)

See also

- [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")

- [`jax.lax.slice()`](jax.lax.slice.html#jax.lax.slice "jax.lax.slice")

- [`jax.lax.dynamic_slice_in_dim()`](jax.lax.dynamic_slice_in_dim.html#jax.lax.dynamic_slice_in_dim "jax.lax.dynamic_slice_in_dim")

- [`jax.lax.dynamic_index_in_dim()`](jax.lax.dynamic_index_in_dim.html#jax.lax.dynamic_index_in_dim "jax.lax.dynamic_index_in_dim")

[](jax.lax.dynamic_index_in_dim.html "previous page")

previous

jax.lax.dynamic_index_in_dim

[](jax.lax.dynamic_slice_in_dim.html "next page")

next

jax.lax.dynamic_slice_in_dim

Contents

- [`dynamic_slice()`](#jax.lax.dynamic_slice)

By The JAX authors

© Copyright 2024, The JAX Authors.\
