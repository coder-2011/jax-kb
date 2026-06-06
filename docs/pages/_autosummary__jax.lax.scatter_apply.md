- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.scatter_apply

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.scatter_apply.rst "Download source file")
-  .pdf

# jax.lax.scatter_apply

## Contents

- [`scatter_apply()`](#jax.lax.scatter_apply)

# jax.lax.scatter_apply[\#](#jax-lax-scatter-apply "Link to this heading")

jax.lax.scatter_apply(*operand*, *scatter_indices*, *func*, *dimension_numbers*, *\**, *update_shape=()*, *indices_are_sorted=False*, *unique_indices=False*, *mode=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L776-L838)[\#](#jax.lax.scatter_apply "Link to this definition")  
Scatter-apply operator.

Wraps [XLA’s Scatter operator](https://www.openxla.org/xla/operation_semantics#scatter), where values from `operand` are replaced with `func(operand)`, with duplicate indices resulting in multiple applications of `func`.

The semantics of scatter are complicated, and its API might change in the future. For most use cases, you should prefer the [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") property on JAX arrays which uses the familiar NumPy indexing syntax.

Note that in the current implementation, `scatter_apply` is not compatible with automatic differentiation.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – an array to which the scatter should be applied

- **scatter_indices** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – an array that gives the indices in operand to which each update in updates should be applied.

- **func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – unary function that will be applied at each index.

- **dimension_numbers** ([*ScatterDimensionNumbers*](../jax.lax.html#jax.lax.ScatterDimensionNumbers "jax.lax.ScatterDimensionNumbers")) – a lax.ScatterDimensionNumbers object that describes how dimensions of operand, start_indices, updates and the output relate.

- **update_shape** (*Shape*) – the shape of the updates at the given indices.

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether scatter_indices is known to be sorted. If true, may improve performance on some backends.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether the elements to be updated in `operand` are guaranteed to not overlap with each other. If true, may improve performance on some backends. JAX does not check this promise: if the updated elements overlap when `unique_indices` is `True` the behavior is undefined.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") *\|* *None*) – how to handle indices that are out of bounds: when set to ‘clip’, indices are clamped so that the slice is within bounds, and when set to ‘fill’ or ‘drop’ out-of-bounds updates are dropped. The behavior for out-of-bounds indices when set to ‘promise_in_bounds’ is implementation-defined.

Returns:  
An array containing the result of applying func to operand at the given indices.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.scatter_add.html "previous page")

previous

jax.lax.scatter_add

[](jax.lax.scatter_max.html "next page")

next

jax.lax.scatter_max

Contents

- [`scatter_apply()`](#jax.lax.scatter_apply)

By The JAX authors

© Copyright 2024, The JAX Authors.\
