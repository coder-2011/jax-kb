- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.cond

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.cond.rst "Download source file")
-  .pdf

# jax.lax.cond

## Contents

- [`cond()`](#jax.lax.cond)

# jax.lax.cond[\#](#jax-lax-cond "Link to this heading")

jax.lax.cond(*pred*, *true_fun*, *false_fun*, *\*operands*, *operand=\<object object\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/conditionals.py#L186-L325)[\#](#jax.lax.cond "Link to this definition")  
Conditionally apply `true_fun` or `false_fun`.

Wraps XLA’s [Conditional](https://www.openxla.org/xla/operation_semantics#conditional) operator.

Provided arguments are correctly typed, `cond()` has equivalent semantics to this Python implementation, where `pred` must be a scalar type:

    def cond(pred, true_fun, false_fun, *operands):
      if pred:
        return true_fun(*operands)
      else:
        return false_fun(*operands)

In contrast with [`jax.lax.select()`](jax.lax.select.html#jax.lax.select "jax.lax.select"), using `cond` indicates that only one of the two branches is executed (up to compiler rewrites and optimizations). However, when transformed with [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") to operate over a batch of predicates, `cond` is converted to [`select()`](jax.lax.select.html#jax.lax.select "jax.lax.select"). Both branches will be traced in all cases (see [Key concepts: tracing](../key-concepts.html#key-concepts-tracing) for a discussion of JAX’s tracing model).

Parameters:  
- **pred** – Boolean scalar type, indicating which branch function to apply.

- **true_fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Function (A -\> B), to be applied if `pred` is True.

- **false_fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Function (A -\> B), to be applied if `pred` is False.

- **operands** – Operands (A) input to either branch depending on `pred`. The type can be a scalar, array, or any pytree (nested Python tuple/list/dict) thereof.

Returns:  
Value (B) of either `true_fun(*operands)` or `false_fun(*operands)`, depending on the value of `pred`. The type can be a scalar, array, or any pytree (nested Python tuple/list/dict) thereof.

[](jax.lax.associative_scan.html "previous page")

previous

jax.lax.associative_scan

[](jax.lax.fori_loop.html "next page")

next

jax.lax.fori_loop

Contents

- [`cond()`](#jax.lax.cond)

By The JAX authors

© Copyright 2024, The JAX Authors.\
