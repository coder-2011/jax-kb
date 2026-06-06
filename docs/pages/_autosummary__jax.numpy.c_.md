- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.c\_

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.c_.rst "Download source file")
-  .pdf

# jax.numpy.c\_

## Contents

- [`c_`](#jax.numpy.c_)

# jax.numpy.c\_[\#](#jax-numpy-c "Link to this heading")

jax.numpy.c\_ *= \<jax.\_src.numpy.index_tricks.CClass object\>*[\#](#jax.numpy.c_ "Link to this definition")  
Concatenate slices, scalars and array-like objects along the last axis.

LAX-backend implementation of [`numpy.c_`](https://numpy.org/doc/stable/reference/generated/numpy.c_.html#numpy.c_ "(in NumPy v2.4)").

See also

`jnp.r_`: Concatenates slices, scalars and array-like objects along the first axis.

Examples

    >>> a = jnp.arange(6).reshape((2,3))
    >>> jnp.c_[a,a]
    Array([[0, 1, 2, 0, 1, 2],
           [3, 4, 5, 3, 4, 5]], dtype=int32)

Use a string directive of the form `"axis:dims:trans1d"` as the first argument to specify concatenation axis, minimum number of dimensions, and the position of the upgraded array’s original dimensions in the resulting array’s shape tuple:

    >>> jnp.c_['0,2', [1,2,3], [4,5,6]]
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

    >>> jnp.c_['0,2,-1', [1,2,3], [4,5,6]]
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)

Use the special directives `"r"` or `"c"` as the first argument on flat inputs to create an array with inputs stacked along the last axis:

    >>> jnp.c_['r',[1,2,3], [4,5,6]]
    Array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)

[](jax.numpy.broadcast_to.html "previous page")

previous

jax.numpy.broadcast_to

[](jax.numpy.can_cast.html "next page")

next

jax.numpy.can_cast

Contents

- [`c_`](#jax.numpy.c_)

By The JAX authors

© Copyright 2024, The JAX Authors.\
