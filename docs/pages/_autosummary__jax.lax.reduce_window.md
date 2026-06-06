- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.reduce_window

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.reduce_window.rst "Download source file")
-  .pdf

# jax.lax.reduce_window

## Contents

- [`reduce_window()`](#jax.lax.reduce_window)

# jax.lax.reduce_window[\#](#jax-lax-reduce-window "Link to this heading")

jax.lax.reduce_window(*operand*, *init_value*, *computation*, *window_dimensions*, *window_strides=None*, *padding='VALID'*, *base_dilation=None*, *window_dilation=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/windowed_reductions.py#L116-L177)[\#](#jax.lax.reduce_window "Link to this definition")  
Reduction over padded windows.

Wraps XLA’s [ReduceWindowWithGeneralPadding](https://www.openxla.org/xla/operation_semantics#reducewindow) operator.

Parameters:  
- **operand** (*Any*) – input array or tree of arrays.

- **init_value** (*Any*) – value or tree of values. Tree structure must match that of `operand`. The values in `init_value` must be scalars.

- **computation** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – callable function over which to reduce. Input and output must be a tree of the same structure as `operand`.

- **window_dimensions** (*core.Shape*) – sequence of integers specifying the window size.

- **window_strides** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional sequence of integers specifying the strides, of the same length as `window_dimensions`. Default (`None`) indicates a unit stride in each window dimension.

- **padding** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – string or sequence of integer tuples specifying the type of padding to use (default: “VALID”). If a string, must be one of “VALID”, “SAME”, or “SAME_LOWER”. See the `jax.lax.padtype_to_pads()` utility.

- **base_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional sequence of integers for base dilation values, of the same length as `window_dimensions`. Default (`None`) indicates unit dilation in each window dimension.

- **window_dilation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional sequence of integers for window dilation values, of the same length as `window_dimensions`. Default (`None`) indicates unit dilation in each window dimension.

Returns:  
A tree of arrays with the same structure as `operand`.

Return type:  
Any

Example

Here is a simple example of a windowed product over pairs in a 1-dimensional array:

    >>> import jax
    >>> x = jax.numpy.arange(10, dtype='float32')
    >>> x
    Array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)

    >>> initial = jax.numpy.float32(1)
    >>> jax.lax.reduce_window(x, initial, jax.lax.mul, window_dimensions=(2,))
    Array([ 0.,  2.,  6., 12., 20., 30., 42., 56., 72.], dtype=float32)

[](jax.lax.reduce_sum.html "previous page")

previous

jax.lax.reduce_sum

[](jax.lax.reduce_xor.html "next page")

next

jax.lax.reduce_xor

Contents

- [`reduce_window()`](#jax.lax.reduce_window)

By The JAX authors

© Copyright 2024, The JAX Authors.\
