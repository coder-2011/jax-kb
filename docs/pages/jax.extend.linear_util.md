- [](index.html)
- [API Reference](jax.html)
- [`jax.extend` module](jax.extend.html)
- `jax.extend.linear_util` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.extend.linear_util.rst "Download source file")
-  .pdf

# jax.extend.linear_util module

# `jax.extend.linear_util` module[\#](#module-jax.extend.linear_util "Link to this heading")

|  |  |
|----|----|
| [`Callable`](_autosummary/jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")() |  |
| [`StoreException`](_autosummary/jax.extend.linear_util.StoreException.html#jax.extend.linear_util.StoreException "jax.extend.linear_util.StoreException") |  |
| [`WrappedFun`](_autosummary/jax.extend.linear_util.WrappedFun.html#jax.extend.linear_util.WrappedFun "jax.extend.linear_util.WrappedFun")(f, f_transformed, transforms, ...) | Represents a function f to which transforms are to be applied. |
| [`cache`](_autosummary/jax.extend.linear_util.cache.html#jax.extend.linear_util.cache "jax.extend.linear_util.cache")(call, \*\[, explain\]) | Memoization decorator for functions taking a WrappedFun as first argument. |
| [`merge_linear_aux`](_autosummary/jax.extend.linear_util.merge_linear_aux.html#jax.extend.linear_util.merge_linear_aux "jax.extend.linear_util.merge_linear_aux")(aux1, aux2) |  |
| [`transformation`](_autosummary/jax.extend.linear_util.transformation.html#jax.extend.linear_util.transformation "jax.extend.linear_util.transformation") |  |
| [`transformation2`](_autosummary/jax.extend.linear_util.transformation2.html#jax.extend.linear_util.transformation2 "jax.extend.linear_util.transformation2") | Adds one more transformation to a WrappedFun. |
| [`transformation_with_aux`](_autosummary/jax.extend.linear_util.transformation_with_aux.html#jax.extend.linear_util.transformation_with_aux "jax.extend.linear_util.transformation_with_aux") |  |
| [`transformation_with_aux2`](_autosummary/jax.extend.linear_util.transformation_with_aux2.html#jax.extend.linear_util.transformation_with_aux2 "jax.extend.linear_util.transformation_with_aux2") | Adds one more transformation with auxiliary output to a WrappedFun. |
| [`wrap_init`](_autosummary/jax.extend.linear_util.wrap_init.html#jax.extend.linear_util.wrap_init "jax.extend.linear_util.wrap_init")(f\[, params, debug_info\]) |  |

[](_autosummary/jax.extend.core.valid_jaxtype.html "previous page")

previous

jax.extend.core.valid_jaxtype

[](_autosummary/jax.extend.linear_util.Callable.html "next page")

next

jax.extend.linear_util.Callable

By The JAX authors

© Copyright 2024, The JAX Authors.\
