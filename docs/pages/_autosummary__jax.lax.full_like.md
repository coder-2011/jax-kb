- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.full_like

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.full_like.rst "Download source file")
-  .pdf

# jax.lax.full_like

## Contents

- [`full_like()`](#jax.lax.full_like)

# jax.lax.full_like[\#](#jax-lax-full-like "Link to this heading")

jax.lax.full_like(*x*, *fill_value*, *dtype=None*, *shape=None*, *sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3793-L3845)[\#](#jax.lax.full_like "Link to this definition")  
Create a full array like np.full based on the example array x.

Parameters:  
- **x** (*ArrayLike* *\|* *DuckTypedArray*) – example array-like, used for shape and dtype information.

- **fill_value** (*ArrayLike*) – a scalar value to fill the entries of the output array.

- **dtype** (*DTypeLike* *\|* *None*) – optional, a dtype parameter for the output ndarray.

- **shape** (*Shape* *\|* *None*) – optional, a shape parameter for the output ndarray.

- **sharding** ([*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – an optional sharding specification for the resulting array. If not specified, the output will have the same sharding as the input, with a few exceptions/limitations in particular: 1. Sharding is not available during tracing, thus this will rely on jit. 2. If x is weakly typed or uncommitted, will use default sharding. 3. Shape is not None and is different from x.shape, default will be used.

Returns:  
An ndarray with the same shape as x with its entries set equal to fill_value, similar to the output of np.full.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.full.html "previous page")

previous

jax.lax.full

[](jax.lax.gather.html "next page")

next

jax.lax.gather

Contents

- [`full_like()`](#jax.lax.full_like)

By The JAX authors

© Copyright 2024, The JAX Authors.\
