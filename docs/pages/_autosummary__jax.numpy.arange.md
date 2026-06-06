- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.arange

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.arange.rst "Download source file")
-  .pdf

# jax.numpy.arange

## Contents

- [`arange()`](#jax.numpy.arange)

# jax.numpy.arange[\#](#jax-numpy-arange "Link to this heading")

jax.numpy.arange(*start*, *stop=None*, *step=None*, *dtype=None*, *\**, *device=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5832-L5918)[\#](#jax.numpy.arange "Link to this definition")  
Create an array of evenly-spaced values.

JAX implementation of [`numpy.arange()`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange "(in NumPy v2.4)"), implemented in terms of [`jax.lax.iota()`](jax.lax.iota.html#jax.lax.iota "jax.lax.iota").

Similar to Python’s `range()` function, this can be called with a few different positional signatures:

- `jnp.arange(stop)`: generate values from 0 to `stop`, stepping by 1.

- `jnp.arange(start,`` ``stop)`: generate values from `start` to `stop`, stepping by 1.

- `jnp.arange(start,`` ``stop,`` ``step)`: generate values from `start` to `stop`, stepping by `step`.

Like with Python’s `range()` function, the starting value is inclusive, and the stop value is exclusive.

Parameters:  
- **start** (*ArrayLike* *\|* *DimSize*) – start of the interval, inclusive.

- **stop** (*ArrayLike* *\|* *DimSize* *\|* *None*) – optional end of the interval, exclusive. If not specified, then `(start,`` ``stop)`` ``=`` ``(0,`` ``start)`

- **step** (*ArrayLike* *\|* *None*) – optional step size for the interval. Default = 1.

- **dtype** (*DTypeLike* *\|* *None*) – optional dtype for the returned array; if not specified it will be determined via type promotion of start, stop, and step.

- **device** (*xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – (optional) [`Device`](jax.Device.html#jax.Device "jax.Device") or [`Sharding`](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") to which the created array will be committed.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – (optional) `NamedSharding` or `P` to which the created array will be committed. Use out_sharding argument, if using explicit sharding ([https://docs.jax.dev/en/latest/parallel.html](https://docs.jax.dev/en/latest/parallel.html))

Returns:  
Array of evenly-spaced values from `start` to `stop`, separated by `step`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Using `arange` with a floating-point `step` argument can lead to unexpected results due to accumulation of floating-point errors, especially with lower-precision data types like `float8_*` and `bfloat16`. To avoid precision errors, consider generating a range of integers, and scaling it to the desired range. For example, instead of this:

    jnp.arange(-1, 1, 0.01, dtype='bfloat16')

it can be more accurate to generate a sequence of integers, and scale them:

    (jnp.arange(-100, 100) * 0.01).astype('bfloat16')

Examples

Single-argument version specifies only the `stop` value:

    >>> jnp.arange(4)
    Array([0, 1, 2, 3], dtype=int32)

Passing a floating-point `stop` value leads to a floating-point result:

    >>> jnp.arange(4.0)
    Array([0., 1., 2., 3.], dtype=float32)

Two-argument version specifies `start` and `stop`, with `step=1`:

    >>> jnp.arange(1, 6)
    Array([1, 2, 3, 4, 5], dtype=int32)

Three-argument version specifies `start`, `stop`, and `step`:

    >>> jnp.arange(0, 2, 0.5)
    Array([0. , 0.5, 1. , 1.5], dtype=float32)

See also

- [`jax.numpy.linspace()`](jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace"): generate a fixed number of evenly-spaced values.

- [`jax.lax.iota()`](jax.lax.iota.html#jax.lax.iota "jax.lax.iota"): directly generate integer sequences in XLA.

[](jax.numpy.apply_over_axes.html "previous page")

previous

jax.numpy.apply_over_axes

[](jax.numpy.arccos.html "next page")

next

jax.numpy.arccos

Contents

- [`arange()`](#jax.numpy.arange)

By The JAX authors

© Copyright 2024, The JAX Authors.\
