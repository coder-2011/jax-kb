- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.with_sharding_constraint

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.with_sharding_constraint.rst "Download source file")
-  .pdf

# jax.lax.with_sharding_constraint

## Contents

- [`with_sharding_constraint()`](#jax.lax.with_sharding_constraint)

# jax.lax.with_sharding_constraint[\#](#jax-lax-with-sharding-constraint "Link to this heading")

jax.lax.with_sharding_constraint(*x*, *shardings*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pjit.py#L2027-L2107)[\#](#jax.lax.with_sharding_constraint "Link to this definition")  
Mechanism to constrain the sharding of an Array inside a jitted computation

This is a strict constraint for the GSPMD partitioner and not a hint. For examples of how to use this function, see [Distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/parallel.html).

Inside of a jitted computation, with_sharding_constraint makes it possible to constrain intermediate values to an uneven sharding. However, if such an unevenly sharded value is output by the jitted computation, it will come out as fully replicated, no matter the sharding annotation given.

Parameters:  
- **x** – PyTree of jax.Arrays which will have their shardings constrained

- **shardings** – PyTree of sharding specifications. Valid values are the same as for the `in_shardings` argument of `jax.experimental.pjit()`.

Returns:  
PyTree of jax.Arrays with specified sharding constraints.

Return type:  
x_with_shardings

[](jax.lax.pbroadcast.html "previous page")

previous

jax.lax.pbroadcast

[](jax.lax.linalg.cholesky.html "next page")

next

jax.lax.linalg.cholesky

Contents

- [`with_sharding_constraint()`](#jax.lax.with_sharding_constraint)

By The JAX authors

© Copyright 2024, The JAX Authors.\
