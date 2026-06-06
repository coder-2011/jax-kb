- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.scatter

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.scatter.rst "Download source file")
-  .pdf

# jax.lax.scatter

## Contents

- [`scatter()`](#jax.lax.scatter)

# jax.lax.scatter[\#](#jax-lax-scatter "Link to this heading")

jax.lax.scatter(*operand*, *scatter_indices*, *updates*, *dimension_numbers*, *\**, *indices_are_sorted=False*, *unique_indices=False*, *mode=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L842-L923)[\#](#jax.lax.scatter "Link to this definition")  
Scatter-update operator.

Wraps [XLA’s Scatter operator](https://www.openxla.org/xla/operation_semantics#scatter), where updates replace values from operand.

If multiple updates are performed to the same index of operand, they may be applied in any order.

[`scatter()`](#jax.lax.scatter "jax.lax.scatter") is a low-level operator with complicated semantics, and most JAX users will never need to call it directly. Instead, you should prefer using [`jax.numpy.ndarray.at()`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") for more familiary NumPy-style indexing syntax.

Parameters:  
- **operand** (*ArrayLike*) – an array to which the scatter should be applied

- **scatter_indices** (*ArrayLike*) – an array that gives the indices in operand to which each update in updates should be applied.

- **updates** (*ArrayLike*) – the updates that should be scattered onto operand.

- **dimension_numbers** ([*ScatterDimensionNumbers*](../jax.lax.html#jax.lax.ScatterDimensionNumbers "jax.lax.ScatterDimensionNumbers")) – a lax.ScatterDimensionNumbers object that describes how dimensions of operand, start_indices, updates and the output relate.

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether scatter_indices is known to be sorted. If true, may improve performance on some backends.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether the elements to be updated in `operand` are guaranteed to not overlap with each other. If true, may improve performance on some backends. JAX does not check this promise: if the updated elements overlap when `unique_indices` is `True` the behavior is undefined.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") *\|* *None*) – how to handle indices that are out of bounds: when set to ‘clip’, indices are clamped so that the slice is within bounds, and when set to ‘fill’ or ‘drop’ out-of-bounds updates are dropped. The behavior for out-of-bounds indices when set to ‘promise_in_bounds’ is implementation-defined.

Returns:  
An array containing the values of operand and the scattered updates.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

As mentioned above, you should basically never use [`scatter()`](#jax.lax.scatter "jax.lax.scatter") directly, and instead perform scatter-style operations using NumPy-style indexing expressions via [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at").

Here is and example of updating entries in an array using [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"), which lowers to an XLA Scatter operation:

    >>> x = jnp.ones(5)
    >>> indices = jnp.array([1, 2, 4])
    >>> values = jnp.array([2.0, 3.0, 4.0])

    >>> x.at[indices].set(values)
    Array([1., 2., 3., 1., 4.], dtype=float32)

This syntax also supports several of the optional arguments to [`scatter()`](#jax.lax.scatter "jax.lax.scatter"), for example:

    >>> x.at[indices].set(values, indices_are_sorted=True, mode='promise_in_bounds')
    Array([1., 2., 3., 1., 4.], dtype=float32)

By comparison, here is the equivalent function call using [`scatter()`](#jax.lax.scatter "jax.lax.scatter") directly, which is not something typical users should ever need to do:

    >>> lax.scatter(x, indices[:, None], values,
    ...             dimension_numbers=lax.ScatterDimensionNumbers(
    ...                 update_window_dims=(),
    ...                 inserted_window_dims=(0,),
    ...                 scatter_dims_to_operand_dims=(0,)),
    ...             indices_are_sorted=True,
    ...             mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
    Array([1., 2., 3., 1., 4.], dtype=float32)

[](jax.lax.scaled_dot.html "previous page")

previous

jax.lax.scaled_dot

[](jax.lax.scatter_add.html "next page")

next

jax.lax.scatter_add

Contents

- [`scatter()`](#jax.lax.scatter)

By The JAX authors

© Copyright 2024, The JAX Authors.\
