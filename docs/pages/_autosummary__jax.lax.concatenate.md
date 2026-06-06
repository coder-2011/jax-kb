- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.concatenate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.concatenate.rst "Download source file")
-  .pdf

# jax.lax.concatenate

## Contents

- [`concatenate()`](#jax.lax.concatenate)

# jax.lax.concatenate[\#](#jax-lax-concatenate "Link to this heading")

jax.lax.concatenate(*operands*, *dimension*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2031-L2054)[\#](#jax.lax.concatenate "Link to this definition")  
Concatenates a sequence of arrays along dimension.

Wraps XLA’s [Concatenate](https://www.openxla.org/xla/operation_semantics#concatenate) operator.

Parameters:  
- **operands** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – a sequence of arrays to concatenate. The arrays must have equal shapes, except in the dimension axis.

- **dimension** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the dimension along which to concatenate the arrays.

Returns:  
An array containing the concatenation.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.composite.html "previous page")

previous

jax.lax.composite

[](jax.lax.conj.html "next page")

next

jax.lax.conj

Contents

- [`concatenate()`](#jax.lax.concatenate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
