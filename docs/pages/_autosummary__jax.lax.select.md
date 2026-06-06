- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.select

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.select.rst "Download source file")
-  .pdf

# jax.lax.select

## Contents

- [`select()`](#jax.lax.select)

# jax.lax.select[\#](#jax-lax-select "Link to this heading")

jax.lax.select(*pred*, *on_true*, *on_false*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3101-L3127)[\#](#jax.lax.select "Link to this definition")  
Selects between two branches based on a boolean predicate.

Wraps XLA’s [Select](https://www.openxla.org/xla/operation_semantics#select) operator.

In general [`select()`](#jax.lax.select "jax.lax.select") leads to evaluation of both branches, although the compiler may elide computations if possible. For a similar function that usually evaluates only a single branch, see [`cond()`](jax.lax.cond.html#jax.lax.cond "jax.lax.cond").

Parameters:  
- **pred** (*ArrayLike*) – boolean array

- **on_true** (*ArrayLike*) – array containing entries to return where `pred` is True. Must have the same shape as `pred`, and the same shape and dtype as `on_false`.

- **on_false** (*ArrayLike*) – array containing entries to return where `pred` is False. Must have the same shape as `pred`, and the same shape and dtype as `on_true`.

Returns:  
array with same shape and dtype as `on_true` and `on_false`.

Return type:  
result

[](jax.lax.scan.html "previous page")

previous

jax.lax.scan

[](jax.lax.select_n.html "next page")

next

jax.lax.select_n

Contents

- [`select()`](#jax.lax.select)

By The JAX authors

© Copyright 2024, The JAX Authors.\
