- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.empty_like

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.empty_like.rst "Download source file")
-  .pdf

# jax.experimental.pallas.empty_like

## Contents

- [`empty_like()`](#jax.experimental.pallas.empty_like)

# jax.experimental.pallas.empty_like[\#](#jax-experimental-pallas-empty-like "Link to this heading")

jax.experimental.pallas.empty_like(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/helpers.py#L38-L54)[\#](#jax.experimental.pallas.empty_like "Link to this definition")  
Create an empty PyTree of possibly uninitialized values.

Parameters:  
**x** ([*object*](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")) – A PyTree with leaves specifying the shape and dtype of the uninitialized object.

Returns:  
A PyTree with the same structure as `x`, but with uninitialized values.

See also

[`jax.lax.empty()`](jax.lax.empty.html#jax.lax.empty "jax.lax.empty")

[](jax.experimental.pallas.empty.html "previous page")

previous

jax.experimental.pallas.empty

[](jax.experimental.pallas.broadcast_to.html "next page")

next

jax.experimental.pallas.broadcast_to

Contents

- [`empty_like()`](#jax.experimental.pallas.empty_like)

By The JAX authors

© Copyright 2024, The JAX Authors.\
