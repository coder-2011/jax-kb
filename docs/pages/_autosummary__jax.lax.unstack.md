- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.unstack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.unstack.rst "Download source file")
-  .pdf

# jax.lax.unstack

## Contents

- [`unstack()`](#jax.lax.unstack)

# jax.lax.unstack[\#](#jax-lax-unstack "Link to this heading")

jax.lax.unstack(*x*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2103-L2125)[\#](#jax.lax.unstack "Link to this definition")  
Unstacks an array along an axis.

Parameters:  
- **x** (*ArrayLike*) – the array to unstack.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to unstack the array.

Returns:  
A tuple of arrays, split along axis.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

Examples

    >>> import jax.numpy as jnp
    >>> from jax import lax
    >>> x = jnp.array([[1, 2], [3, 4]])
    >>> lax.unstack(x, axis=0)
    (Array([1, 2], dtype=int32), Array([3, 4], dtype=int32))
    >>> lax.unstack(x, axis=1)
    (Array([1, 3], dtype=int32), Array([2, 4], dtype=int32))

[](jax.lax.transpose.html "previous page")

previous

jax.lax.transpose

[](jax.lax.zeta.html "next page")

next

jax.lax.zeta

Contents

- [`unstack()`](#jax.lax.unstack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
