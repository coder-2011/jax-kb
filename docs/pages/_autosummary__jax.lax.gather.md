- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.gather

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.gather.rst "Download source file")
-  .pdf

# jax.lax.gather

## Contents

- [`gather()`](#jax.lax.gather)

# jax.lax.gather[\#](#jax-lax-gather "Link to this heading")

jax.lax.gather(*operand*, *start_indices*, *dimension_numbers*, *slice_sizes*, *\**, *unique_indices=False*, *indices_are_sorted=False*, *mode=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/slicing.py#L324-L432)[\#](#jax.lax.gather "Link to this definition")  
Gather operator.

Wraps [XLA’s Gather operator](https://www.openxla.org/xla/operation_semantics#gather).

[`gather()`](#jax.lax.gather "jax.lax.gather") is a low-level operator with complicated semantics, and most JAX users will never need to call it directly. Instead, you should prefer using [Numpy-style indexing](https://numpy.org/doc/stable/reference/arrays.indexing.html), and/or [`jax.numpy.ndarray.at()`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at"), perhaps in combination with [`jax.vmap()`](jax.vmap.html#jax.vmap "jax.vmap").

Parameters:  
- **operand** (*ArrayLike*) – an array from which slices should be taken

- **start_indices** (*ArrayLike*) – the indices at which slices should be taken

- **dimension_numbers** ([*GatherDimensionNumbers*](../jax.lax.html#jax.lax.GatherDimensionNumbers "jax.lax.GatherDimensionNumbers")) – a lax.GatherDimensionNumbers object that describes how dimensions of operand, start_indices and the output relate.

- **slice_sizes** (*Shape*) – the size of each slice. Must be a sequence of non-negative integers with length equal to ndim(operand).

- **indices_are_sorted** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether indices is known to be sorted. If true, may improve performance on some backends.

- **unique_indices** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – whether the elements gathered from `operand` are guaranteed not to overlap with each other. If `True`, this may improve performance on some backends. JAX does not check this promise: if the elements overlap the behavior is undefined.

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* [*GatherScatterMode*](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") *\|* *None*) – how to handle indices that are out of bounds: when set to `'clip'`, indices are clamped so that the slice is within bounds, and when set to `'fill'` or `'drop'` gather returns a slice full of `fill_value` for the affected slice. The behavior for out-of-bounds indices when set to `'promise_in_bounds'` is implementation-defined.

- **fill_value** – the fill value to return for out-of-bounds slices when mode is `'fill'`. Ignored otherwise. Defaults to `NaN` for inexact types, the largest negative value for signed types, the largest positive value for unsigned types, and `True` for booleans.

Returns:  
An array containing the gather output.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

As mentioned above, you should basically never use [`gather()`](#jax.lax.gather "jax.lax.gather") directly, and instead use NumPy-style indexing expressions to gather values from arrays.

For example, here is how you can extract values at particular indices using straightforward indexing semantics, which will lower to XLA’s Gather operator:

    >>> import jax.numpy as jnp
    >>> x = jnp.array([10, 11, 12])
    >>> indices = jnp.array([0, 1, 1, 2, 2, 2])

    >>> x[indices]
    Array([10, 11, 11, 12, 12, 12], dtype=int32)

For control over settings like `indices_are_sorted`, `unique_indices`, `mode`, and `fill_value`, you can use the [`jax.numpy.ndarray.at`](jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") syntax:

    >>> x.at[indices].get(indices_are_sorted=True, mode="promise_in_bounds")
    Array([10, 11, 11, 12, 12, 12], dtype=int32)

By comparison, here is the equivalent function call using [`gather()`](#jax.lax.gather "jax.lax.gather") directly, which is not something typical users should ever need to do:

    >>> from jax import lax
    >>> lax.gather(x, indices[:, None], slice_sizes=(1,),
    ...            dimension_numbers=lax.GatherDimensionNumbers(
    ...                offset_dims=(),
    ...                collapsed_slice_dims=(0,),
    ...                start_index_map=(0,)),
    ...            indices_are_sorted=True,
    ...            mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS)
    Array([10, 11, 11, 12, 12, 12], dtype=int32)

[](jax.lax.full_like.html "previous page")

previous

jax.lax.full_like

[](jax.lax.ge.html "next page")

next

jax.lax.ge

Contents

- [`gather()`](#jax.lax.gather)

By The JAX authors

© Copyright 2024, The JAX Authors.\
