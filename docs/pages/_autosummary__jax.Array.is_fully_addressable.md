- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.is_fully_addressable

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.is_fully_addressable.rst "Download source file")
-  .pdf

# jax.Array.is_fully_addressable

## Contents

- [`Array.is_fully_addressable`](#jax.Array.is_fully_addressable)

# jax.Array.is_fully_addressable[\#](#jax-array-is-fully-addressable "Link to this heading")

*property* Array.is_fully_addressable*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/basearray.py#L113-L126)[\#](#jax.Array.is_fully_addressable "Link to this definition")  
Is this Array fully addressable?

A jax.Array is fully addressable if the current process can address all of the devices named in the `Sharding`. `is_fully_addressable` is equivalent to “is_local” in multi-process JAX.

Note that fully replicated is not equal to fully addressable i.e. a jax.Array which is fully replicated can span across multiple hosts and is not fully addressable.

[](jax.Array.imag.html "previous page")

previous

jax.Array.imag

[](jax.Array.is_fully_replicated.html "next page")

next

jax.Array.is_fully_replicated

Contents

- [`Array.is_fully_addressable`](#jax.Array.is_fully_addressable)

By The JAX authors

© Copyright 2024, The JAX Authors.\
