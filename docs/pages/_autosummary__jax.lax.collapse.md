- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.collapse

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.collapse.rst "Download source file")
-  .pdf

# jax.lax.collapse

## Contents

- [`collapse()`](#jax.lax.collapse)

# jax.lax.collapse[\#](#jax-lax-collapse "Link to this heading")

jax.lax.collapse(*operand*, *start_dimension*, *stop_dimension=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3859-L3885)[\#](#jax.lax.collapse "Link to this definition")  
Collapses dimensions of an array into a single dimension.

For example, if `operand` is an array with shape `[2,`` ``3,`` ``4]`, `collapse(operand,`` ``0,`` ``2).shape`` ``==`` ``[6,`` ``4]`. The elements of the collapsed dimension are laid out major-to-minor, i.e., with the lowest-numbered dimension as the slowest varying dimension.

Parameters:  
- **operand** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – an input array.

- **start_dimension** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the start of the dimensions to collapse (inclusive).

- **stop_dimension** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – the end of the dimensions to collapse (exclusive). Pass None to collapse all the dimensions after start.

Returns:  
An array where dimensions `[start_dimension,`` ``stop_dimension)` have been collapsed (raveled) into a single dimension.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.clz.html "previous page")

previous

jax.lax.clz

[](jax.lax.complex.html "next page")

next

jax.lax.complex

Contents

- [`collapse()`](#jax.lax.collapse)

By The JAX authors

© Copyright 2024, The JAX Authors.\
