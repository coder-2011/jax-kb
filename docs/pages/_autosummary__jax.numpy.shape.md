- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.shape

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.shape.rst "Download source file")
-  .pdf

# jax.numpy.shape

## Contents

- [`shape()`](#jax.numpy.shape)

# jax.numpy.shape[\#](#jax-numpy-shape "Link to this heading")

jax.numpy.shape(*a*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/util.py#L374-L417)[\#](#jax.numpy.shape "Link to this definition")  
Return the shape an array.

JAX implementation of [`numpy.shape()`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html#numpy.shape "(in NumPy v2.4)"). Unlike `np.shape`, this function raises a [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") if the input is a collection such as a list or tuple.

Parameters:  
**a** (*ArrayLike* *\|* *SupportsShape*) – array-like object, or any object with a `shape` attribute.

Returns:  
An tuple of integers representing the shape of `a`.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), …\]

Examples

Shape for arrays:

    >>> x = jnp.arange(10)
    >>> jnp.shape(x)
    (10,)
    >>> y = jnp.ones((2, 3))
    >>> jnp.shape(y)
    (2, 3)

This also works for scalars:

    >>> jnp.shape(3.14)
    ()

For arrays, this can also be accessed via the [`jax.Array.shape`](jax.Array.shape.html#jax.Array.shape "jax.Array.shape") property:

    >>> x.shape
    (10,)

[](jax.numpy.setxor1d.html "previous page")

previous

jax.numpy.setxor1d

[](jax.numpy.sign.html "next page")

next

jax.numpy.sign

Contents

- [`shape()`](#jax.numpy.shape)

By The JAX authors

© Copyright 2024, The JAX Authors.\
