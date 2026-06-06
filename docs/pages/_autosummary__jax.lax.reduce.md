- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.reduce

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.reduce.rst "Download source file")
-  .pdf

# jax.lax.reduce

## Contents

- [`reduce()`](#jax.lax.reduce)

# jax.lax.reduce[\#](#jax-lax-reduce "Link to this heading")

jax.lax.reduce(*operands*, *init_values*, *computation*, *dimensions*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3179-L3227)[\#](#jax.lax.reduce "Link to this definition")  
Wraps XLA’s [Reduce](https://www.openxla.org/xla/operation_semantics#reduce) operator.

`init_values` and `computation` together must form a [monoid](https://en.wikipedia.org/wiki/Monoid) for correctness. That is `init_values` must be an identity of `computation`, and `computation` must be associative. XLA may exploit both of these properties during code generation; if either is violated the result is undefined.

`init_values` must consist of scalars.

Parameters:  
- **operands** (*Any*)

- **init_values** (*Any*)

- **computation** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Any,* *Any\],* *Any\]*)

- **dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Return type:  
Any

[](jax.lax.reciprocal.html "previous page")

previous

jax.lax.reciprocal

[](jax.lax.reduce_and.html "next page")

next

jax.lax.reduce_and

Contents

- [`reduce()`](#jax.lax.reduce)

By The JAX authors

© Copyright 2024, The JAX Authors.\
