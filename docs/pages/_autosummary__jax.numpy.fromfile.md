- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fromfile

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fromfile.rst "Download source file")
-  .pdf

# jax.numpy.fromfile

## Contents

- [`fromfile()`](#jax.numpy.fromfile)

# jax.numpy.fromfile[\#](#jax-numpy-fromfile "Link to this heading")

jax.numpy.fromfile(*\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5502-L5519)[\#](#jax.numpy.fromfile "Link to this definition")  
Unimplemented JAX wrapper for jnp.fromfile.

This function is left deliberately unimplemented because it may be non-pure and thus unsafe for use with JIT and other JAX transformations. Consider using `jnp.asarray(np.fromfile(...))` instead, although care should be taken if `np.fromfile` is used within jax transformations because of its potential side-effect of consuming the file object; for more information see [Common Gotchas: Pure Functions](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions).

[](jax.numpy.frombuffer.html "previous page")

previous

jax.numpy.frombuffer

[](jax.numpy.fromfunction.html "next page")

next

jax.numpy.fromfunction

Contents

- [`fromfile()`](#jax.numpy.fromfile)

By The JAX authors

© Copyright 2024, The JAX Authors.\
