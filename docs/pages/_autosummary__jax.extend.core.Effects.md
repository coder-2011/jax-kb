- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.core` module](../jax.extend.core.html)
- jax.extend.core.Effects

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.core.Effects.rst "Download source file")
-  .pdf

# jax.extend.core.Effects

## Contents

- [`Effects`](#jax.extend.core.Effects)

# jax.extend.core.Effects[\#](#jax-extend-core-effects "Link to this heading")

jax.extend.core.Effects[\#](#jax.extend.core.Effects "Link to this definition")  
A set is a finite, iterable container.

This class provides concrete generic implementations of all methods except for \_\_contains\_\_, \_\_iter\_\_ and \_\_len\_\_.

To override the comparisons (presumably for speed, as the semantics are fixed), redefine \_\_le\_\_ and \_\_ge\_\_, then the other operations will automatically follow suit.

alias of [`Set`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Set "(in Python v3.14)")\[[`Effect`](jax.extend.core.Effect.html#jax.extend.core.Effect "jax._src.effects.Effect")\]

[](jax.extend.core.Effect.html "previous page")

previous

jax.extend.core.Effect

[](jax.extend.core.InconclusiveDimensionOperation.html "next page")

next

jax.extend.core.InconclusiveDimensionOperation

Contents

- [`Effects`](#jax.extend.core.Effects)

By The JAX authors

© Copyright 2024, The JAX Authors.\
