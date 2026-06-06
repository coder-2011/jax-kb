- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ogrid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ogrid.rst "Download source file")
-  .pdf

# jax.numpy.ogrid

## Contents

- [`ogrid`](#jax.numpy.ogrid)

# jax.numpy.ogrid[\#](#jax-numpy-ogrid "Link to this heading")

jax.numpy.ogrid *= \<jax.\_src.numpy.index_tricks.\_Ogrid object\>*[\#](#jax.numpy.ogrid "Link to this definition")  
Return open multi-dimensional “meshgrid”.

LAX-backend implementation of [`numpy.ogrid`](https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html#numpy.ogrid "(in NumPy v2.4)"). This is a convenience wrapper for functionality provided by [`jax.numpy.meshgrid()`](jax.numpy.meshgrid.html#jax.numpy.meshgrid "jax.numpy.meshgrid") with `sparse=True`.

See also

jnp.mgrid: dense version of jnp.ogrid

Examples

Pass `[start:stop:step]` to generate values similar to [`jax.numpy.arange()`](jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange"):

    >>> jnp.ogrid[0:4:1]
    Array([0, 1, 2, 3], dtype=int32)

Passing an imaginary step generates values similar to [`jax.numpy.linspace()`](jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace"):

    >>> jnp.ogrid[0:1:4j]
    Array([0.        , 0.33333334, 0.6666667 , 1.        ], dtype=float32)

Multiple slices can be used to create sparse grids of indices:

    >>> jnp.ogrid[:2, :3]
    [Array([[0],
            [1]], dtype=int32),
     Array([[0, 1, 2]], dtype=int32)]

[](jax.numpy.object_.html "previous page")

previous

jax.numpy.object\_

[](jax.numpy.ones.html "next page")

next

jax.numpy.ones

Contents

- [`ogrid`](#jax.numpy.ogrid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
