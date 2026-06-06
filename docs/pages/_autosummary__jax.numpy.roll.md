- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.roll

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.roll.rst "Download source file")
-  .pdf

# jax.numpy.roll

## Contents

- [`roll()`](#jax.numpy.roll)

# jax.numpy.roll[\#](#jax-numpy-roll "Link to this heading")

jax.numpy.roll(*a*, *shift*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L8497-L8548)[\#](#jax.numpy.roll "Link to this definition")  
Roll the elements of an array along a specified axis.

JAX implementation of [`numpy.roll()`](https://numpy.org/doc/stable/reference/generated/numpy.roll.html#numpy.roll "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array.

- **shift** (*ArrayLike* *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the number of positions to shift the specified axis. If an integer, all axes are shifted by the same amount. If a tuple, the shift for each axis is specified individually.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – the axis or axes to roll. If `None`, the array is flattened, shifted, and then reshaped to its original shape.

Returns:  
A copy of `a` with elements rolled along the specified axis or axes.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.rollaxis()`](jax.numpy.rollaxis.html#jax.numpy.rollaxis "jax.numpy.rollaxis"): roll the specified axis to a given position.

Examples

    >>> a = jnp.array([0, 1, 2, 3, 4, 5])
    >>> jnp.roll(a, 2)
    Array([4, 5, 0, 1, 2, 3], dtype=int32)

Roll elements along a specific axis:

    >>> a = jnp.array([[ 0,  1,  2,  3],
    ...                [ 4,  5,  6,  7],
    ...                [ 8,  9, 10, 11]])
    >>> jnp.roll(a, 1, axis=0)
    Array([[ 8,  9, 10, 11],
           [ 0,  1,  2,  3],
           [ 4,  5,  6,  7]], dtype=int32)
    >>> jnp.roll(a, [2, 3], axis=[0, 1])
    Array([[ 5,  6,  7,  4],
           [ 9, 10, 11,  8],
           [ 1,  2,  3,  0]], dtype=int32)

[](jax.numpy.rint.html "previous page")

previous

jax.numpy.rint

[](jax.numpy.rollaxis.html "next page")

next

jax.numpy.rollaxis

Contents

- [`roll()`](#jax.numpy.roll)

By The JAX authors

© Copyright 2024, The JAX Authors.\
