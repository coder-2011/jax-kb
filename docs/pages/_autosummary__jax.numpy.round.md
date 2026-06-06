- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.round

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.round.rst "Download source file")
-  .pdf

# jax.numpy.round

## Contents

- [`round()`](#jax.numpy.round)

# jax.numpy.round[\#](#jax-numpy-round "Link to this heading")

jax.numpy.round(*a*, *decimals=0*, *out=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3413-L3486)[\#](#jax.numpy.round "Link to this definition")  
Round input evenly to the given number of decimals.

JAX implementation of [`numpy.round()`](https://numpy.org/doc/stable/reference/generated/numpy.round.html#numpy.round "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array or scalar.

- **decimals** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – int, default=0. Number of decimal points to which the input needs to be rounded. It must be specified statically. Not implemented for `decimals`` ``<`` ``0`.

- **out** (*None*) – Unused by JAX.

Returns:  
An array containing the rounded values to the specified `decimals` with same shape and dtype as `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

`jnp.round` rounds to the nearest even integer for the values exactly halfway between rounded decimal values.

See also

- [`jax.numpy.floor()`](jax.numpy.floor.html#jax.numpy.floor "jax.numpy.floor"): Rounds the input to the nearest integer downwards.

- [`jax.numpy.ceil()`](jax.numpy.ceil.html#jax.numpy.ceil "jax.numpy.ceil"): Rounds the input to the nearest integer upwards.

- `jax.numpy.fix()` and :func:numpy.trunc\`: Rounds the input to the nearest integer towards zero.

Examples

    >>> x = jnp.array([1.532, 3.267, 6.149])
    >>> jnp.round(x)
    Array([2., 3., 6.], dtype=float32)
    >>> jnp.round(x, decimals=2)
    Array([1.53, 3.27, 6.15], dtype=float32)

For values exactly halfway between rounded values:

    >>> x1 = jnp.array([10.5, 21.5, 12.5, 31.5])
    >>> jnp.round(x1)
    Array([10., 22., 12., 32.], dtype=float32)

[](jax.numpy.rot90.html "previous page")

previous

jax.numpy.rot90

[](jax.numpy.s_.html "next page")

next

jax.numpy.s\_

Contents

- [`round()`](#jax.numpy.round)

By The JAX authors

© Copyright 2024, The JAX Authors.\
