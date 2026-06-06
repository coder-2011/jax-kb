- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.size

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.size.rst "Download source file")
-  .pdf

# jax.numpy.size

## Contents

- [`size()`](#jax.numpy.size)

# jax.numpy.size[\#](#jax-numpy-size "Link to this heading")

jax.numpy.size(*a*, *axis=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/util.py#L419-L468)[\#](#jax.numpy.size "Link to this definition")  
Return number of elements along a given axis.

JAX implementation of [`numpy.size()`](https://numpy.org/doc/stable/reference/generated/numpy.size.html#numpy.size "(in NumPy v2.4)"). Unlike `np.size`, this function raises a [`TypeError`](https://docs.python.org/3/library/exceptions.html#TypeError "(in Python v3.14)") if the input is a collection such as a list or tuple.

Parameters:  
- **a** (*ArrayLike* *\|* *SupportsSize* *\|* *SupportsShape*) – array-like object, or any object with a `size` attribute when `axis` is not specified, or with a `shape` attribute when `axis` is specified.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional integer or sequence of integers indicating which axis or axes to count elements along. `None` (the default) returns the total number of elements.

Returns:  
An integer specifying the number of elements in `a`.

Return type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

Examples

Size for arrays:

    >>> x = jnp.arange(10)
    >>> jnp.size(x)
    10
    >>> y = jnp.ones((2, 3))
    >>> jnp.size(y)
    6
    >>> jnp.size(y, axis=1)
    3
    >>> jnp.size(y, axis=(1,))
    3
    >>> jnp.size(y, axis=(0, 1))
    6

This also works for scalars:

    >>> jnp.size(3.14)
    1

For arrays, this can also be accessed via the [`jax.Array.size`](jax.Array.size.html#jax.Array.size "jax.Array.size") property:

    >>> y.size
    6

[](jax.numpy.sinh.html "previous page")

previous

jax.numpy.sinh

[](jax.numpy.sort.html "next page")

next

jax.numpy.sort

Contents

- [`size()`](#jax.numpy.size)

By The JAX authors

© Copyright 2024, The JAX Authors.\
