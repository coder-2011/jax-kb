- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.linalg.tensorsolve

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.linalg.tensorsolve.rst "Download source file")
-  .pdf

# jax.numpy.linalg.tensorsolve

## Contents

- [`tensorsolve()`](#jax.numpy.linalg.tensorsolve)

# jax.numpy.linalg.tensorsolve[\#](#jax-numpy-linalg-tensorsolve "Link to this heading")

jax.numpy.linalg.tensorsolve(*a*, *b*, *axes=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/linalg.py#L2063-L2110)[\#](#jax.numpy.linalg.tensorsolve "Link to this definition")  
Solve the tensor equation a x = b for x.

JAX implementation of [`numpy.linalg.tensorsolve()`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensorsolve.html#numpy.linalg.tensorsolve "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array. After reordering via `axes` (see below), shape must be `(*b.shape,`` ``*x.shape)`.

- **b** (*ArrayLike*) – right-hand-side array.

- **axes** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]* *\|* *None*) – optional tuple specifying axes of `a` that should be moved to the end

Returns:  
array x such that after reordering of axes of `a`, `tensordot(a,`` ``x,`` ``x.ndim)` is equivalent to `b`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.linalg.tensordot()`](jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot")

- [`jax.numpy.linalg.tensorinv()`](jax.numpy.linalg.tensorinv.html#jax.numpy.linalg.tensorinv "jax.numpy.linalg.tensorinv")

Examples

    >>> key1, key2 = jax.random.split(jax.random.key(8675309))
    >>> a = jax.random.normal(key1, shape=(2, 2, 4))
    >>> b = jax.random.normal(key2, shape=(2, 2))
    >>> x = jnp.linalg.tensorsolve(a, b)
    >>> x.shape
    (4,)

Now show that `x` can be used to reconstruct `b` using [`tensordot()`](jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot"):

    >>> b_reconstructed = jnp.linalg.tensordot(a, x, axes=x.ndim)
    >>> jnp.allclose(b, b_reconstructed)
    Array(True, dtype=bool)

[](jax.numpy.linalg.tensorinv.html "previous page")

previous

jax.numpy.linalg.tensorinv

[](jax.numpy.linalg.trace.html "next page")

next

jax.numpy.linalg.trace

Contents

- [`tensorsolve()`](#jax.numpy.linalg.tensorsolve)

By The JAX authors

© Copyright 2024, The JAX Authors.\
