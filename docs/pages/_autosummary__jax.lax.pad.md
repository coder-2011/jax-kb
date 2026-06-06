- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.pad

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.pad.rst "Download source file")
-  .pdf

# jax.lax.pad

## Contents

- [`pad()`](#jax.lax.pad)

# jax.lax.pad[\#](#jax-lax-pad "Link to this heading")

jax.lax.pad(*operand*, *padding_value*, *padding_config*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3034-L3093)[\#](#jax.lax.pad "Link to this definition")  
Applies low, high, and/or interior padding to an array.

Wraps XLA’s [Pad](https://www.openxla.org/xla/operation_semantics#pad) operator.

Parameters:  
- **operand** (*ArrayLike*) – an array to be padded.

- **padding_value** (*ArrayLike*) – the value to be inserted as padding. Must have the same dtype as `operand`.

- **padding_config** (*Sequence\[*[*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]\]*) – a sequence of `(low,`` ``high,`` ``interior)` tuples of integers, giving the amount of low, high, and interior (dilation) padding to insert in each dimension. Negative values for `low` and `high` are allowed and remove elements from the edges of the array.

Returns:  
The `operand` array with padding value `padding_value` inserted in each dimension according to the `padding_config`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> from jax import lax
    >>> import jax.numpy as jnp

Pad a 1-dimensional array with zeros, We’ll specify two zeros in front and three at the end:

    >>> x = jnp.array([1, 2, 3, 4])
    >>> lax.pad(x, 0, [(2, 3, 0)])
    Array([0, 0, 1, 2, 3, 4, 0, 0, 0], dtype=int32)

Pad a 1-dimensional array with *interior* zeros; i.e. insert a single zero between each value:

    >>> lax.pad(x, 0, [(0, 0, 1)])
    Array([1, 0, 2, 0, 3, 0, 4], dtype=int32)

Pad a 2-dimensional array with the value `-1` at front and end, with a pad size of 2 in each dimension:

    >>> x = jnp.array([[1, 2, 3],
    ...                [4, 5, 6]])
    >>> lax.pad(x, -1, [(2, 2, 0), (2, 2, 0)])
    Array([[-1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1],
           [-1, -1,  1,  2,  3, -1, -1],
           [-1, -1,  4,  5,  6, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1, -1]], dtype=int32)

Use negative padding to remove elements from the edges of an array:

    >>> x = jnp.array([1, 2, 3, 4, 5], dtype=jnp.int32)
    >>> lax.pad(x, 0, [(-1, -2, 0)])
    Array([2, 3], dtype=int32)

[](jax.lax.optimization_barrier.html "previous page")

previous

jax.lax.optimization_barrier

[](jax.lax.platform_dependent.html "next page")

next

jax.lax.platform_dependent

Contents

- [`pad()`](#jax.lax.pad)

By The JAX authors

© Copyright 2024, The JAX Authors.\
