- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.full

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.full.rst "Download source file")
-  .pdf

# jax.lax.full

## Contents

- [`full()`](#jax.lax.full)

# jax.lax.full[\#](#jax-lax-full "Link to this heading")

jax.lax.full(*shape*, *fill_value*, *dtype=None*, *\**, *sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3588-L3626)[\#](#jax.lax.full "Link to this definition")  
Returns an array of shape filled with fill_value.

Parameters:  
- **shape** (*Shape*) – sequence of integers, describing the shape of the output array.

- **fill_value** (*ArrayLike*) – the value to fill the new array with.

- **dtype** (*DTypeLike* *\|* *None*) – the type of the output array, or None. If not None, fill_value will be cast to dtype.

- **sharding** ([*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – an optional sharding specification for the resulting array, note, sharding will currently be ignored in jitted mode, this might change in the future.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.floor.html "previous page")

previous

jax.lax.floor

[](jax.lax.full_like.html "next page")

next

jax.lax.full_like

Contents

- [`full()`](#jax.lax.full)

By The JAX authors

© Copyright 2024, The JAX Authors.\
