- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.linear_util` module](../jax.extend.linear_util.html)
- jax.extend.linear_util.cache

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.linear_util.cache.rst "Download source file")
-  .pdf

# jax.extend.linear_util.cache

## Contents

- [`cache()`](#jax.extend.linear_util.cache)

# jax.extend.linear_util.cache[\#](#jax-extend-linear-util-cache "Link to this heading")

jax.extend.linear_util.cache(*call*, *\**, *explain=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/linear_util.py#L436-L480)[\#](#jax.extend.linear_util.cache "Link to this definition")  
Memoization decorator for functions taking a WrappedFun as first argument.

Parameters:  
- **call** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – a Python callable that takes a WrappedFun as its first argument. The underlying transforms and params on the WrappedFun are used as part of the memoization cache key.

- **explain** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[*[*WrappedFun*](jax.extend.linear_util.WrappedFun.html#jax.extend.linear_util.WrappedFun "jax.extend.linear_util.WrappedFun")*,* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*,* [*dict*](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*,* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")*\],* *None\]* *\|* *None*) – a function that is invoked upon cache misses to log an explanation of the miss. Invoked with (fun, is_cache_first_use, cache, key, elapsed_sec).

Returns:  
A memoized version of `call`.

[](jax.extend.linear_util.WrappedFun.html "previous page")

previous

jax.extend.linear_util.WrappedFun

[](jax.extend.linear_util.merge_linear_aux.html "next page")

next

jax.extend.linear_util.merge_linear_aux

Contents

- [`cache()`](#jax.extend.linear_util.cache)

By The JAX authors

© Copyright 2024, The JAX Authors.\
