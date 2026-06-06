- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.committed

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.committed.rst "Download source file")
-  .pdf

# jax.Array.committed

## Contents

- [`Array.committed`](#jax.Array.committed)

# jax.Array.committed[\#](#jax-array-committed "Link to this heading")

*property* Array.committed*: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")*[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/basearray.py#L137-L160)[\#](#jax.Array.committed "Link to this definition")  
Whether the array is committed or not.

An array is committed when it is explicitly placed on device(s) via JAX APIs. For example, `jax.device_put(np.arange(8),`` ``jax.devices()[0])` is committed to device 0. While `jax.device_put(np.arange(8))` is uncommitted and will be placed on the default device.

Computations involving some committed inputs will happen on the committed device(s) and the result will be committed on the same device(s). Invoking an operation on arguments that are committed to different device(s) will raise an error.

Examples

    >>> a = jax.device_put(np.arange(8), jax.devices()[0])
    >>> b = jax.device_put(np.arange(8), jax.devices()[1])
    >>> a + b  
    Traceback (most recent call last):
      ...
    ValueError: Received incompatible devices for jitted computation.

[](jax.Array.compress.html "previous page")

previous

jax.Array.compress

[](jax.Array.conj.html "next page")

next

jax.Array.conj

Contents

- [`Array.committed`](#jax.Array.committed)

By The JAX authors

© Copyright 2024, The JAX Authors.\
