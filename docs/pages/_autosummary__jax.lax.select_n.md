- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.select_n

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.select_n.rst "Download source file")
-  .pdf

# jax.lax.select_n

## Contents

- [`select_n()`](#jax.lax.select_n)

# jax.lax.select_n[\#](#jax-lax-select-n "Link to this heading")

jax.lax.select_n(*which*, *\*cases*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3128-L3153)[\#](#jax.lax.select_n "Link to this definition")  
Selects array values from multiple cases.

Generalizes XLA’s [Select](https://www.openxla.org/xla/operation_semantics#select) operator. Unlike XLA’s version, the operator is variadic and can select from many cases using an integer pred.

Parameters:  
- **which** (*ArrayLike*) – determines which case should be returned. Must be an array containing either a boolean or integer values. May either be a scalar or have shape matching `cases`. For each array element, the value of `which` determines which of `cases` is taken. `which` must be in the range `[0`` ``..`` ``len(cases))`; for values outside that range the behavior is implementation-defined.

- **\*cases** (*ArrayLike*) – a non-empty list of array cases. All must have equal dtypes and equal shapes.

Returns:  
An array with shape and dtype equal to the cases, whose values are chosen according to `which`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.select.html "previous page")

previous

jax.lax.select

[](jax.lax.switch.html "next page")

next

jax.lax.switch

Contents

- [`select_n()`](#jax.lax.select_n)

By The JAX authors

© Copyright 2024, The JAX Authors.\
