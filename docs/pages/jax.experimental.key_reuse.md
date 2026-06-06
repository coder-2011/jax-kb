- [](index.html)
- [API Reference](jax.html)
- [`jax.experimental` module](jax.experimental.html)
- `jax.experimental.key_reuse` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.experimental.key_reuse.rst "Download source file")
-  .pdf

# jax.experimental.key_reuse module

## Contents

- [Experimental Key Reuse Checking](#experimental-key-reuse-checking)

# `jax.experimental.key_reuse` module[\#](#module-jax.experimental.key_reuse "Link to this heading")

## Experimental Key Reuse Checking[\#](#experimental-key-reuse-checking "Link to this heading")

This module contains **experimental** functionality for detecting reuse of random keys within JAX programs. It is under active development and the APIs here are likely to change. The usage below requires JAX version 0.4.26 or newer.

Key reuse checking can be enabled using the `jax_debug_key_reuse` configuration. This can be set globally using:

    >>> jax.config.update('jax_debug_key_reuse', True)  

Or it can be enabled locally with the `jax.debug_key_reuse()` context manager. When enabled, using the same key twice will result in a [`KeyReuseError`](errors.html#jax.errors.KeyReuseError "jax.errors.KeyReuseError"):

    >>> import jax
    >>> with jax.debug_key_reuse(True):
    ...   key = jax.random.key(0)
    ...   val1 = jax.random.normal(key)
    ...   val2 = jax.random.normal(key)  
    Traceback (most recent call last):
     ...
    KeyReuseError: Previously-consumed key passed to jit-compiled function at index 0

The key reuse checker is currently experimental, but in the future we will likely enable it by default.

[](jax.experimental.jet.html "previous page")

previous

`jax.experimental.jet` module

[](jax.experimental.mesh_utils.html "next page")

next

`jax.experimental.mesh_utils` module

Contents

- [Experimental Key Reuse Checking](#experimental-key-reuse-checking)

By The JAX authors

© Copyright 2024, The JAX Authors.\
