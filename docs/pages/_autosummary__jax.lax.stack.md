- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.stack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.stack.rst "Download source file")
-  .pdf

# jax.lax.stack

## Contents

- [`stack()`](#jax.lax.stack)

# jax.lax.stack[\#](#jax-lax-stack "Link to this heading")

jax.lax.stack(*operands*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2076-L2102)[\#](#jax.lax.stack "Link to this definition")  
Joins a sequence of arrays along a new axis.

Parameters:  
- **operands** (*Sequence\[ArrayLike\]*) – a sequence of arrays to stack. All arrays must have the same shape.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to stack the arrays.

Returns:  
An array containing the stacked operands.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> x = jnp.array([1, 2])
    >>> y = jnp.array([3, 4])
    >>> lax.stack([x, y], axis=0)
    Array([[1, 2],
           [3, 4]], dtype=int32)
    >>> lax.stack([x, y], axis=1)
    Array([[1, 3],
           [2, 4]], dtype=int32)

[](jax.lax.squeeze.html "previous page")

previous

jax.lax.squeeze

[](jax.lax.stage.html "next page")

next

jax.lax.stage

Contents

- [`stack()`](#jax.lax.stack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
