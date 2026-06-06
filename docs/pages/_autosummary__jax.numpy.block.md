- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.block

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.block.rst "Download source file")
-  .pdf

# jax.numpy.block

## Contents

- [`block()`](#jax.numpy.block)

# jax.numpy.block[\#](#jax-numpy-block "Link to this heading")

jax.numpy.block(*arrays*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4988-L5061)[\#](#jax.numpy.block "Link to this definition")  
Create an array from a list of blocks.

JAX implementation of [`numpy.block()`](https://numpy.org/doc/stable/reference/generated/numpy.block.html#numpy.block "(in NumPy v2.4)").

Parameters:  
**arrays** (*ArrayLike* *\|* [*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[ArrayLike\]*) – an array, or nested list of arrays which will be concatenated together to form the final array.

Returns:  
a single array constructed from the inputs.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`concatenate()`](jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate"), [`concat()`](jax.numpy.concat.html#jax.numpy.concat "jax.numpy.concat"): concatenate arrays along an existing axis.

- [`stack()`](jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack"), [`vstack()`](jax.numpy.vstack.html#jax.numpy.vstack "jax.numpy.vstack"), [`hstack()`](jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack"), [`dstack()`](jax.numpy.dstack.html#jax.numpy.dstack "jax.numpy.dstack") concatenate arrays along a new axis.

Examples

consider these blocks:

    >>> zeros = jnp.zeros((2, 2))
    >>> ones = jnp.ones((2, 2))
    >>> twos = jnp.full((2, 2), 2)
    >>> threes = jnp.full((2, 2), 3)

Passing a single array to [`block()`](#jax.numpy.block "jax.numpy.block") returns the array:

    >>> jnp.block(zeros)
    Array([[0., 0.],
           [0., 0.]], dtype=float32)

Passing a simple list of arrays concatenates them along the last axis:

    >>> jnp.block([zeros, ones])
    Array([[0., 0., 1., 1.],
           [0., 0., 1., 1.]], dtype=float32)

Passing a doubly-nested list of arrays concatenates the inner list along the last axis, and the outer list along the second-to-last axis:

    >>> jnp.block([[zeros, ones],
    ...            [twos, threes]])
    Array([[0., 0., 1., 1.],
           [0., 0., 1., 1.],
           [2., 2., 3., 3.],
           [2., 2., 3., 3.]], dtype=float32)

Note that blocks need not align in all dimensions, though the size along the axis of concatenation must match. For example, this is valid because after the inner, horizontal concatenation, the resulting blocks have a valid shape for the outer, vertical concatenation.

    >>> a = jnp.zeros((2, 1))
    >>> b = jnp.ones((2, 3))
    >>> c = jnp.full((1, 2), 2)
    >>> d = jnp.full((1, 2), 3)
    >>> jnp.block([[a, b], [c, d]])
    Array([[0., 1., 1., 1.],
           [0., 1., 1., 1.],
           [2., 2., 3., 3.]], dtype=float32)

Note also that this logic generalizes to blocks in 3 or more dimensions. Here’s a 3-dimensional block-wise array:

    >>> x = jnp.arange(6).reshape((1, 2, 3))
    >>> blocks = [[[x for i in range(3)] for j in range(4)] for k in range(5)]
    >>> jnp.block(blocks).shape
    (5, 8, 9)

[](jax.numpy.blackman.html "previous page")

previous

jax.numpy.blackman

[](jax.numpy.bool_.html "next page")

next

jax.numpy.bool\_

Contents

- [`block()`](#jax.numpy.block)

By The JAX authors

© Copyright 2024, The JAX Authors.\
