- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.unstack

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.unstack.rst "Download source file")
-  .pdf

# jax.numpy.unstack

## Contents

- [`unstack()`](#jax.numpy.unstack)

# jax.numpy.unstack[\#](#jax-numpy-unstack "Link to this heading")

jax.numpy.unstack(*x*, */*, *\**, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4423-L4457)[\#](#jax.numpy.unstack "Link to this definition")  
Unstack an array along an axis.

JAX implementation of [`array_api.unstack()`](https://data-apis.org/array-api/2023.12/API_specification/generated/array_api.unstack.html#array_api.unstack "(in Python array API standard)").

Parameters:  
- **x** (*ArrayLike*) – array to unstack. Must have `x.ndim`` ``>=`` ``1`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer axis along which to unstack. Must satisfy `-x.ndim`` ``<=`` ``axis`` ``<`` ``x.ndim`.

Returns:  
tuple of unstacked arrays.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

See also

- [`jax.numpy.stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack"): inverse of `unstack`

- [`jax.numpy.split()`](jax.numpy.split.html#jax.numpy.split "jax.numpy.split"): split array into batches along an axis.

Examples

    >>> arr = jnp.array([[1, 2, 3],
    ...                  [4, 5, 6]])
    >>> arrs = jnp.unstack(arr)
    >>> print(*arrs)
    [1 2 3] [4 5 6]

[`stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack") provides the inverse of this:

    >>> jnp.stack(arrs)
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)

[](jax.numpy.unravel_index.html "previous page")

previous

jax.numpy.unravel_index

[](jax.numpy.unsignedinteger.html "next page")

next

jax.numpy.unsignedinteger

Contents

- [`unstack()`](#jax.numpy.unstack)

By The JAX authors

© Copyright 2024, The JAX Authors.\
