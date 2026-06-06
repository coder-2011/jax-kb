- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.mgrid

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.mgrid.rst "Download source file")
-  .pdf

# jax.numpy.mgrid

## Contents

- [`mgrid`](#jax.numpy.mgrid)

# jax.numpy.mgrid[\#](#jax-numpy-mgrid "Link to this heading")

jax.numpy.mgrid *= \<jax.\_src.numpy.index_tricks.\_Mgrid object\>*[\#](#jax.numpy.mgrid "Link to this definition")  
Return dense multi-dimensional “meshgrid”.

LAX-backend implementation of [`numpy.mgrid`](https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html#numpy.mgrid "(in NumPy v2.4)"). This is a convenience wrapper for functionality provided by [`jax.numpy.meshgrid()`](jax.numpy.meshgrid.html#jax.numpy.meshgrid "jax.numpy.meshgrid") with `sparse=False`.

See also

jnp.ogrid: open/sparse version of jnp.mgrid

Examples

Pass `[start:stop:step]` to generate values similar to [`jax.numpy.arange()`](jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange"):

    >>> jnp.mgrid[0:4:1]
    Array([0, 1, 2, 3], dtype=int32)

Passing an imaginary step generates values similar to [`jax.numpy.linspace()`](jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace"):

    >>> jnp.mgrid[0:1:4j]
    Array([0.        , 0.33333334, 0.6666667 , 1.        ], dtype=float32)

Multiple slices can be used to create broadcasted grids of indices:

    >>> jnp.mgrid[:2, :3]
    Array([[[0, 0, 0],
            [1, 1, 1]],
           [[0, 1, 2],
            [0, 1, 2]]], dtype=int32)

[](jax.numpy.meshgrid.html "previous page")

previous

jax.numpy.meshgrid

[](jax.numpy.min.html "next page")

next

jax.numpy.min

Contents

- [`mgrid`](#jax.numpy.mgrid)

By The JAX authors

© Copyright 2024, The JAX Authors.\
