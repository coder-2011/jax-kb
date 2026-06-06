- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.scatter_add

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.scatter_add.rst "Download source file")
-  .pdf

# jax.lax.scatter_add

## Contents

- [`scatter_add()`](#jax.lax.scatter_add)

# jax.lax.scatter_add[\#](#jax-lax-scatter-add "Link to this heading")

jax.lax.scatter_add(*operand*, *scatter_indices*, *updates*, *dimension_numbers*, *\**, *indices_are_sorted=False*, *unique_indices=False*, *mode=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L473-L558)[\#](#jax.lax.scatter_add "Link to this definition")  
Scatter-add operator.

Wraps [XLA’s Scatter operator](https://www.openxla.org/xla/operation_semantics#scatter), where addition is used to combine updates and values from operand.

The semantics of scatter are complicated, and its API might change in the future. For most use cases, you should prefer the [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") property on JAX arrays which uses the familiar NumPy indexing syntax.

Parameters:  
- **operand** (*ArrayLike*) – an array to which the scatter should be applied

- **scatter_indices** (*ArrayLike*) – an array that gives the indices in operand to which each update in updates should be applied.

- **updates** (*ArrayLike*) – the updates that should be scattered onto operand.

- **dimension_numbers** ([*ScatterDimensionNumbers*](../jax.lax.html#jax.lax.ScatterDimensionNumbers "jax.lax.ScatterDimensionNumbers")) – a lax.ScatterDimensionNumbers object that describes how dimensions of operand, scatter_indices, updates and the output relate.

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether scatter_indices is known to be sorted. If true, may improve performance on some backends.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether the elements to be updated in `operand` are guaranteed to not overlap with each other. If true, may improve performance on some backends. JAX does not check this promise: if the updated elements overlap when `unique_indices` is `True` the behavior is undefined.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") *\|* *None*) – how to handle indices that are out of bounds: when set to ‘clip’, indices are clamped so that the slice is within bounds, and when set to ‘fill’ or ‘drop’ out-of-bounds updates are dropped. The behavior for out-of-bounds indices when set to ‘promise_in_bounds’ is implementation-defined.

Returns:  
An array containing the sum of operand and the scattered updates.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

As mentioned above, you should basically never use [`scatter_add()`](#jax.lax.scatter_add "jax.lax.scatter_add") directly, and instead perform scatter-style operations using NumPy-style indexing expressions via [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at").

Here is and example of updating entries in an array using [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"), which lowers to an XLA Scatter operation:

    >>> x = jnp.ones(5)
    >>> indices = jnp.array([1, 2, 4])
    >>> values = jnp.array([2.0, 3.0, 4.0])

    >>> x.at[indices].add(values)
    Array([1., 3., 4., 1., 5.], dtype=float32)

This syntax also supports several of the optional arguments to [`scatter_add()`](#jax.lax.scatter_add "jax.lax.scatter_add"), for example:

    >>> x.at[indices].add(values, indices_are_sorted=True,
    ...                   mode='promise_in_bounds')
    Array([1., 3., 4., 1., 5.], dtype=float32)

By comparison, here is the equivalent function call using [`scatter_add()`](#jax.lax.scatter_add "jax.lax.scatter_add") directly, which is not something typical users should ever need to do:

    >>> lax.scatter_add(x, indices[:, None], values,
    ...                 dimension_numbers=lax.ScatterDimensionNumbers(
    ...                     update_window_dims=(),
    ...                     inserted_window_dims=(0,),
    ...                     scatter_dims_to_operand_dims=(0,)),
    ...                 indices_are_sorted=True,
    ...                 mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
    Array([1., 3., 4., 1., 5.], dtype=float32)

[](jax.lax.scatter.html "previous page")

previous

jax.lax.scatter

[](jax.lax.scatter_apply.html "next page")

next

jax.lax.scatter_apply

Contents

- [`scatter_add()`](#jax.lax.scatter_add)

By The JAX authors

© Copyright 2024, The JAX Authors.\
