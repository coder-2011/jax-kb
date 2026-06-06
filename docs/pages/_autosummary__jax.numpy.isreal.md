- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isreal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isreal.rst "Download source file")
-  .pdf

# jax.numpy.isreal

## Contents

- [`isreal()`](#jax.numpy.isreal)

# jax.numpy.isreal[\#](#jax-numpy-isreal "Link to this heading")

jax.numpy.isreal(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1482-L1505)[\#](#jax.numpy.isreal "Link to this definition")  
Return boolean array showing where the input is real.

JAX implementation of [`numpy.isreal()`](https://numpy.org/doc/stable/reference/generated/numpy.isreal.html#numpy.isreal "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – input array to check.

Returns:  
A new array containing boolean values indicating real elements.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.iscomplex()`](jax.numpy.iscomplex.html#jax.numpy.iscomplex "jax.numpy.iscomplex")

- [`jax.numpy.isrealobj()`](jax.numpy.isrealobj.html#jax.numpy.isrealobj "jax.numpy.isrealobj")

Examples

    >>> jnp.isreal(jnp.array([False, 0j, 1, 2.1, 1+2j]))
    Array([ True,  True,  True,  True, False], dtype=bool)

[](jax.numpy.isposinf.html "previous page")

previous

jax.numpy.isposinf

[](jax.numpy.isrealobj.html "next page")

next

jax.numpy.isrealobj

Contents

- [`isreal()`](#jax.numpy.isreal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
