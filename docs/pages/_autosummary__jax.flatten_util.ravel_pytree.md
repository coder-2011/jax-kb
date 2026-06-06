- [](../index.html)
- [API Reference](../jax.html)
- [`jax.flatten_util` module](../jax.flatten_util.html)
- jax.flatten_util.ravel_pytree

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.flatten_util.ravel_pytree.rst "Download source file")
-  .pdf

# jax.flatten_util.ravel_pytree

## Contents

- [`ravel_pytree()`](#jax.flatten_util.ravel_pytree)

# jax.flatten_util.ravel_pytree[\#](#jax-flatten-util-ravel-pytree "Link to this heading")

jax.flatten_util.ravel_pytree(*pytree*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/flatten_util.py#L30-L52)[\#](#jax.flatten_util.ravel_pytree "Link to this definition")  
Ravel (flatten) a pytree of arrays down to a 1D array.

Parameters:  
**pytree** ([*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")) – a pytree of arrays and scalars to ravel.

Returns:  
A pair where the first element is a 1D array representing the flattened and concatenated leaf values, with dtype determined by promoting the dtypes of leaf values, and the second element is a callable for unflattening a 1D vector of the same length back to a pytree of the same structure as the input `pytree`. If the input pytree is empty (i.e. has no leaves) then as a convention a 1D empty array of dtype float32 is returned in the first component of the output.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[*Array*](jax.Array.html#jax.Array "jax.Array"), [*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")\[\[[*Array*](jax.Array.html#jax.Array "jax.Array")\], [*Any*](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")\]\]

For details on dtype promotion, see [https://docs.jax.dev/en/latest/type_promotion.html](https://docs.jax.dev/en/latest/type_promotion.html).

[](../jax.flatten_util.html "previous page")

previous

`jax.flatten_util` module

[](../jax.image.html "next page")

next

`jax.image` module

Contents

- [`ravel_pytree()`](#jax.flatten_util.ravel_pytree)

By The JAX authors

© Copyright 2024, The JAX Authors.\
