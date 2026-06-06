- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.union1d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.union1d.rst "Download source file")
-  .pdf

# jax.numpy.union1d

## Contents

- [`union1d()`](#jax.numpy.union1d)

# jax.numpy.union1d[\#](#jax-numpy-union1d "Link to this heading")

jax.numpy.union1d(*ar1*, *ar2*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L183-L254)[\#](#jax.numpy.union1d "Link to this definition")  
Compute the set union of two 1D arrays.

JAX implementation of [`numpy.union1d()`](https://numpy.org/doc/stable/reference/generated/numpy.union1d.html#numpy.union1d "(in NumPy v2.4)").

Because the size of the output of `union1d` is data-dependent, the function is not typically compatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations. The JAX version adds the optional `size` argument which must be specified statically for `jnp.union1d` to be used in such contexts.

Parameters:  
- **ar1** (*ArrayLike*) – first array of elements to be unioned.

- **ar2** (*ArrayLike*) – second array of elements to be unioned

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, return only the first `size` sorted elements. If there are fewer elements than `size` indicates, the return value will be padded with `fill_value`.

- **fill_value** (*ArrayLike* *\|* *None*) – when `size` is specified and there are fewer than the indicated number of elements, fill the remaining entries `fill_value`. Defaults to the minimum value.

Returns:  
an array containing the union of elements in the input array.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.intersect1d()`](jax.numpy.intersect1d.html#jax.numpy.intersect1d "jax.numpy.intersect1d"): the set intersection of two 1D arrays.

- [`jax.numpy.setxor1d()`](jax.numpy.setxor1d.html#jax.numpy.setxor1d "jax.numpy.setxor1d"): the set XOR of two 1D arrays.

- [`jax.numpy.setdiff1d()`](jax.numpy.setdiff1d.html#jax.numpy.setdiff1d "jax.numpy.setdiff1d"): the set difference of two 1D arrays.

Examples

Computing the union of two arrays:

    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.union1d(ar1, ar2)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

Because the output shape is dynamic, this will fail under [`jit()`](jax.jit.html#jax.jit "jax.jit") and other transformations:

    >>> jax.jit(jnp.union1d)(ar1, ar2)  
    Traceback (most recent call last):
       ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[4].
    The error occurred while tracing the function union1d at /Users/vanderplas/github/jax-ml/jax/jax/_src/numpy/setops.py:101 for jit. This concrete value was not available in Python because it depends on the value of the argument ar1.

In order to ensure statically-known output shapes, you can pass a static `size` argument:

    >>> jit_union1d = jax.jit(jnp.union1d, static_argnames=['size'])
    >>> jit_union1d(ar1, ar2, size=6)
    Array([1, 2, 3, 4, 5, 6], dtype=int32)

If `size` is too small, the union is truncated:

    >>> jit_union1d(ar1, ar2, size=4)
    Array([1, 2, 3, 4], dtype=int32)

If `size` is too large, then the output is padded with `fill_value`:

    >>> jit_union1d(ar1, ar2, size=8, fill_value=0)
    Array([1, 2, 3, 4, 5, 6, 0, 0], dtype=int32)

[](jax.numpy.uint8.html "previous page")

previous

jax.numpy.uint8

[](jax.numpy.unique.html "next page")

next

jax.numpy.unique

Contents

- [`union1d()`](#jax.numpy.union1d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
