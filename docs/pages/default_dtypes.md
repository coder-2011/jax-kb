- [](index.html)
- [Notes](notes.html)
- Default dtypes and the X64 flag

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/default_dtypes.md "Download source file")
-  .pdf

# Default dtypes and the X64 flag

## Contents

- [Default setting: 32-bits everywhere](#default-setting-32-bits-everywhere)
- [The X64 flag: enabling 64-bit values](#the-x64-flag-enabling-64-bit-values)

# Default dtypes and the X64 flag[\#](#default-dtypes-and-the-x64-flag "Link to this heading")

JAX strives to meet the needs of a range of numerical computing practitioners, who sometimes have conflicting preferences. When it comes to default dtypes, there are two different camps:

- Classic scientific computing practitioners (i.e. users of tools like [`numpy`](https://numpy.org/doc/stable/reference/index.html#module-numpy "(in NumPy v2.4)") or [`scipy`](https://scipy.github.io/devdocs/index.html#module-scipy "(in SciPy v1.19.0.dev)")) tend to value accuracy of computations foremost: such users would prefer that computations default to the **widest available representation**: e.g. floating point values should default to `float64`, integers to `int64`, etc.

- AI researchers (i.e. folks implementing and training neural networks) tend to value speed over accuracy, to the point where they have developed special data types like [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) and others which deliberately discard the least significant bits in order to speed up computation. For these users, the mere presence of a float64 value in their computation can lead to programs that are slow at best, and incompatible with their hardware at worst! These users would prefer that computations default to `float32` or `int32`.

The main mechanism JAX offers for this is the `jax_enable_x64` flag, which controls whether 64-bit values can be created at all. By default this flag is set to `False` (serving the needs of AI researchers and practitioners), but can be set to `True` by users who value accuracy over computational speed.

## Default setting: 32-bits everywhere[\#](#default-setting-32-bits-everywhere "Link to this heading")

By default `jax_enable_x64` is set to False, and so [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") array creation functions will default to returning 32-bit values.

For example:

    >>> import jax.numpy as jnp

    >>> jnp.arange(5)
    Array([0, 1, 2, 3, 4], dtype=int32)

    >>> jnp.zeros(5)
    Array([0., 0., 0., 0., 0.], dtype=float32)

    >>> jnp.ones(5, dtype=int)
    Array([1, 1, 1, 1, 1], dtype=int32)

Beyond defaults, because 64-bit values can be so poisonous to AI workflows, having this flag set to False prevents you from creating 64-bit arrays at all! For example:

    >>> jnp.arange(5, dtype='float64')  
    UserWarning: Explicitly requested dtype float64 requested in arange is not available, and will be 
    truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the 
    JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
    Array([0., 1., 2., 3., 4.], dtype=float32)

## The X64 flag: enabling 64-bit values[\#](#the-x64-flag-enabling-64-bit-values "Link to this heading")

To work in the “other mode” where functions default to producing 64-bit values, you can set the `jax_enable_x64` flag to `True`:

    import jax
    import jax.numpy as jnp

    jax.config.update('jax_enable_x64', True)

    print(repr(jnp.arange(5)))
    print(repr(jnp.zeros(5)))
    print(repr(jnp.ones(5, dtype=int)))

    Array([0, 1, 2, 3, 4], dtype=int64)
    Array([0., 0., 0., 0., 0.], dtype=float64)
    Array([1, 1, 1, 1, 1], dtype=int64)

The X64 configuration can also be set via the `JAX_ENABLE_X64` shell environment variable, for example:

    $ JAX_ENABLE_X64=1 python main.py

The X64 flag is intended as a **global setting** that should have one value for your whole program, set at the top of your main file. A common feature request is for the flag to be contextually configurable (e.g. enabling X64 just for one section of a long program): this turns out to be difficult to implement within JAX’s programming model, where code execution may happen in a different context than code compilation. There is ongoing work exploring the feasibility of relaxing this constraint, so stay tuned!

[](type_promotion.html "previous page")

previous

Type promotion semantics

[](pallas/index.html "next page")

next

Pallas: a JAX kernel language

Contents

- [Default setting: 32-bits everywhere](#default-setting-32-bits-everywhere)
- [The X64 flag: enabling 64-bit values](#the-x64-flag-enabling-64-bit-values)

By The JAX authors

© Copyright 2024, The JAX Authors.\
