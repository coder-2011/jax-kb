- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fromiter

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fromiter.rst "Download source file")
-  .pdf

# jax.numpy.fromiter

## Contents

- [`fromiter()`](#jax.numpy.fromiter)

# jax.numpy.fromiter[\#](#jax-numpy-fromiter "Link to this heading")

jax.numpy.fromiter(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5521-L5538)[\#](#jax.numpy.fromiter "Link to this definition")  
Unimplemented JAX wrapper for jnp.fromiter.

This function is left deliberately unimplemented because it may be non-pure and thus unsafe for use with JIT and other JAX transformations. Consider using `jnp.asarray(np.fromiter(...))` instead, although care should be taken if `np.fromiter` is used within jax transformations because of its potential side-effect of consuming the iterable object; for more information see [Common Gotchas: Pure Functions](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).

[](jax.numpy.fromfunction.html "previous page")

previous

jax.numpy.fromfunction

[](jax.numpy.frompyfunc.html "next page")

next

jax.numpy.frompyfunc

Contents

- [`fromiter()`](#jax.numpy.fromiter)

By The JAX authors

© Copyright 2024, The JAX Authors.\
