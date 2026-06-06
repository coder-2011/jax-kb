- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.clone

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.clone.rst "Download source file")
-  .pdf

# jax.random.clone

## Contents

- [`clone()`](#jax.random.clone)

# jax.random.clone[\#](#jax-random-clone "Link to this heading")

jax.random.clone(*key*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L3397-L3413)[\#](#jax.random.clone "Link to this definition")  
Clone a key for reuse

Outside the context of key reuse checking (see [`jax.experimental.key_reuse`](../jax.experimental.key_reuse.html#module-jax.experimental.key_reuse "jax.experimental.key_reuse")) this function operates as an identity.

Examples

    >>> import jax
    >>> key = jax.random.key(0)
    >>> data = jax.random.uniform(key)
    >>> cloned_key = jax.random.clone(key)
    >>> same_data = jax.random.uniform(cloned_key)
    >>> assert data == same_data

[](jax.random.split.html "previous page")

previous

jax.random.split

[](jax.random.PRNGKey.html "next page")

next

jax.random.PRNGKey

Contents

- [`clone()`](#jax.random.clone)

By The JAX authors

© Copyright 2024, The JAX Authors.\
