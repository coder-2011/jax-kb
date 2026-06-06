- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.iterable

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.iterable.rst "Download source file")
-  .pdf

# jax.numpy.iterable

## Contents

- [`iterable()`](#jax.numpy.iterable)

# jax.numpy.iterable[\#](#jax-numpy-iterable "Link to this heading")

jax.numpy.iterable(*y*)[\#](#jax.numpy.iterable "Link to this definition")  
Check whether or not an object can be iterated over.

Parameters:  
**y** ([*object*](https://docs.python.org/3/library/functions.html#object "(in Python v3.14)")) – Input object.

Returns:  
**b** – Return `True` if the object has an iterator method or is a sequence and `False` otherwise.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

Examples

    >>> import numpy as np
    >>> np.iterable([1, 2, 3])
    True
    >>> np.iterable(2)
    False

Notes

In most cases, the results of `np.iterable(obj)` are consistent with `isinstance(obj,`` ``collections.abc.Iterable)`. One notable exception is the treatment of 0-dimensional arrays:

    >>> from collections.abc import Iterable
    >>> a = np.array(1.0)  # 0-dimensional numpy array
    >>> isinstance(a, Iterable)
    True
    >>> np.iterable(a)
    False

[](jax.numpy.issubdtype.html "previous page")

previous

jax.numpy.issubdtype

[](jax.numpy.ix_.html "next page")

next

jax.numpy.ix\_

Contents

- [`iterable()`](#jax.numpy.iterable)

By The JAX authors

© Copyright 2024, The JAX Authors.\
