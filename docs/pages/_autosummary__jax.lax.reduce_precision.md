- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.reduce_precision

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.reduce_precision.rst "Download source file")
-  .pdf

# jax.lax.reduce_precision

## Contents

- [`reduce_precision()`](#jax.lax.reduce_precision)

# jax.lax.reduce_precision[\#](#jax-lax-reduce-precision "Link to this heading")

jax.lax.reduce_precision(*operand*, *exponent_bits*, *mantissa_bits*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3753-L3766)[\#](#jax.lax.reduce_precision "Link to this definition")  
Wraps XLA’s [ReducePrecision](https://www.openxla.org/xla/operation_semantics#reduceprecision) operator.

Parameters:  
- **operand** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* *ArrayLike*)

- **exponent_bits** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **mantissa_bits** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.reduce_or.html "previous page")

previous

jax.lax.reduce_or

[](jax.lax.reduce_prod.html "next page")

next

jax.lax.reduce_prod

Contents

- [`reduce_precision()`](#jax.lax.reduce_precision)

By The JAX authors

© Copyright 2024, The JAX Authors.\
