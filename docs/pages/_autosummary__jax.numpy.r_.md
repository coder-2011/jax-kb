- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.r\_

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.r_.rst "Download source file")
-  .pdf

# jax.numpy.r\_

## Contents

- [`r_`](#jax.numpy.r_)

# jax.numpy.r\_[\#](#jax-numpy-r "Link to this heading")

jax.numpy.r\_ *= \<jax.\_src.numpy.index_tricks.RClass object\>*[\#](#jax.numpy.r_ "Link to this definition")  
Concatenate slices, scalars and array-like objects along the first axis.

LAX-backend implementation of [`numpy.r_`](https://numpy.org/doc/stable/reference/generated/numpy.r_.html#numpy.r_ "(in NumPy v2.4)").

See also

`jnp.c_`: Concatenates slices, scalars and array-like objects along the last axis.

Examples

Passing slices in the form `[start:stop:step]` generates `jnp.arange` objects:

    >>> jnp.r_[-1:5:1, 0, 0, jnp.array([1,2,3])]
    Array([-1,  0,  1,  2,  3,  4,  0,  0,  1,  2,  3], dtype=int32)

An imaginary value for `step` will create a `jnp.linspace` object instead, which includes the right endpoint:

    >>> jnp.r_[-1:1:6j, 0, jnp.array([1,2,3])]  
    Array([-1. , -0.6, -0.2,  0.2,  0.6,  1. ,  0. ,  1. ,  2. ,  3. ],      dtype=float32)

Use a string directive of the form `"axis,dims,trans1d"` as the first argument to specify concatenation axis, minimum number of dimensions, and the position of the upgraded array’s original dimensions in the resulting array’s shape tuple:

    >>> jnp.r_['0,2', [1,2,3], [4,5,6]] # concatenate along first axis, 2D output
    Array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)

    >>> jnp.r_['0,2,0', [1,2,3], [4,5,6]] # push last input axis to the front
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

Negative values for `trans1d` offset the last axis towards the start of the shape tuple:

    >>> jnp.r_['0,2,-2', [1,2,3], [4,5,6]]
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

Use the special directives `"r"` or `"c"` as the first argument on flat inputs to create an array with an extra row or column axis, respectively:

    >>> jnp.r_['r',[1,2,3], [4,5,6]]
    Array([[1, 2, 3, 4, 5, 6]], dtype=int32)

    >>> jnp.r_['c',[1,2,3], [4,5,6]]
    Array([[1],
           [2],
           [3],
           [4],
           [5],
           [6]], dtype=int32)

For higher-dimensional inputs (`dim`` ``>=`` ``2`), both directives `"r"` and `"c"` give the same result.

[](jax.numpy.quantile.html "previous page")

previous

jax.numpy.quantile

[](jax.numpy.rad2deg.html "next page")

next

jax.numpy.rad2deg

Contents

- [`r_`](#jax.numpy.r_)

By The JAX authors

© Copyright 2024, The JAX Authors.\
