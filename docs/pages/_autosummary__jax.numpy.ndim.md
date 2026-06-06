- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.ndim

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.ndim.rst "Download source file")
-  .pdf

# jax.numpy.ndim

## Contents

- [`ndim()`](#jax.numpy.ndim)

# jax.numpy.ndim[\#](#jax-numpy-ndim "Link to this heading")

jax.numpy.ndim(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/util.py#L329-L372)[\#](#jax.numpy.ndim "Link to this definition")  
Return the number of dimensions of an array.

JAX implementation of [`numpy.ndim()`](https://numpy.org/doc/stable/reference/generated/numpy.ndim.html#numpy.ndim "(in NumPy v2.4)"). Unlike `np.ndim`, this function raises a [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") if the input is a collection such as a list or tuple.

Parameters:  
**a** (*ArrayLike* *\|* *SupportsNdim*) – array-like object, or any object with an `ndim` attribute.

Returns:  
An integer specifying the number of dimensions of `a`.

Return type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

Examples

Number of dimensions for arrays:

    >>> x = jnp.arange(10)
    >>> jnp.ndim(x)
    1
    >>> y = jnp.ones((2, 3))
    >>> jnp.ndim(y)
    2

This also works for scalars:

    >>> jnp.ndim(3.14)
    0

For arrays, this can also be accessed via the [`jax.Array.ndim`](jax.Array.ndim.html#jax.Array.ndim "jax.Array.ndim") property:

    >>> x.ndim
    1

[](jax.numpy.ndarray.html "previous page")

previous

jax.numpy.ndarray

[](jax.numpy.negative.html "next page")

next

jax.numpy.negative

Contents

- [`ndim()`](#jax.numpy.ndim)

By The JAX authors

© Copyright 2024, The JAX Authors.\
