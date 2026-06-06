- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.scatter_mul

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.scatter_mul.rst "Download source file")
-  .pdf

# jax.lax.scatter_mul

## Contents

- [`scatter_mul()`](#jax.lax.scatter_mul)

# jax.lax.scatter_mul[\#](#jax-lax-scatter-mul "Link to this heading")

jax.lax.scatter_mul(*operand*, *scatter_indices*, *updates*, *dimension_numbers*, *\**, *indices_are_sorted=False*, *unique_indices=False*, *mode=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L623-L672)[\#](#jax.lax.scatter_mul "Link to this definition")  
Scatter-multiply operator.

Wraps [XLA’s Scatter operator](https://www.openxla.org/xla/operation_semantics#scatter), where multiplication is used to combine updates and values from operand.

The semantics of scatter are complicated, and its API might change in the future. For most use cases, you should prefer the [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") property on JAX arrays which uses the familiar NumPy indexing syntax.

Parameters:  
- **operand** (*ArrayLike*) – an array to which the scatter should be applied

- **scatter_indices** (*ArrayLike*) – an array that gives the indices in operand to which each update in updates should be applied.

- **updates** (*ArrayLike*) – the updates that should be scattered onto operand.

- **dimension_numbers** ([*ScatterDimensionNumbers*](../jax.lax.html#jax.lax.ScatterDimensionNumbers "jax.lax.ScatterDimensionNumbers")) – a lax.ScatterDimensionNumbers object that describes how dimensions of operand, start_indices, updates and the output relate.

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether scatter_indices is known to be sorted. If true, may improve performance on some backends.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether the elements to be updated in `operand` are guaranteed to not overlap with each other. If true, may improve performance on some backends. JAX does not check this promise: if the updated elements overlap when `unique_indices` is `True` the behavior is undefined.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") *\|* *None*) – how to handle indices that are out of bounds: when set to ‘clip’, indices are clamped so that the slice is within bounds, and when set to ‘fill’ or ‘drop’ out-of-bounds updates are dropped. The behavior for out-of-bounds indices when set to ‘promise_in_bounds’ is implementation-defined.

Returns:  
An array containing the product of operand and the scattered updates.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.scatter_min.html "previous page")

previous

jax.lax.scatter_min

[](jax.lax.scatter_sub.html "next page")

next

jax.lax.scatter_sub

Contents

- [`scatter_mul()`](#jax.lax.scatter_mul)

By The JAX authors

© Copyright 2024, The JAX Authors.\
