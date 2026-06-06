- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.setdiff1d

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.setdiff1d.rst "Download source file")
-  .pdf

# jax.numpy.setdiff1d

## Contents

- [`setdiff1d()`](#jax.numpy.setdiff1d)

# jax.numpy.setdiff1d[\#](#jax-numpy-setdiff1d "Link to this heading")

jax.numpy.setdiff1d(*ar1*, *ar2*, *assume_unique=False*, *\**, *size=None*, *fill_value=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/setops.py#L92-L181)[\#](#jax.numpy.setdiff1d "Link to this definition")  
Compute the set difference of two 1D arrays.

JAX implementation of [`numpy.setdiff1d()`](https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html#numpy.setdiff1d "(in NumPy v2.4)").

Because the size of the output of `setdiff1d` is data-dependent, the function is not typically compatible with [`jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations. The JAX version adds the optional `size` argument which must be specified statically for `jnp.setdiff1d` to be used in such contexts.

Parameters:  
- **ar1** (*ArrayLike*) – first array of elements to be differenced.

- **ar2** (*ArrayLike*) – second array of elements to be differenced.

- **assume_unique** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, assume the input arrays contain unique values. This allows a more efficient implementation, but if `assume_unique` is True and the input arrays contain duplicates, the behavior is undefined. default: False.

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – if specified, return only the first `size` sorted elements. If there are fewer elements than `size` indicates, the return value will be padded with `fill_value`.

- **fill_value** (*ArrayLike* *\|* *None*) – when `size` is specified and there are fewer than the indicated number of elements, fill the remaining entries `fill_value`. Defaults to the minimum value.

Returns:  
i.e. the elements in `ar1` that are not contained in `ar2`.

Return type:  
an array containing the set difference of elements in the input array

See also

- [`jax.numpy.intersect1d()`](jax.numpy.intersect1d.html#jax.numpy.intersect1d "jax.numpy.intersect1d"): the set intersection of two 1D arrays.

- [`jax.numpy.setxor1d()`](jax.numpy.setxor1d.html#jax.numpy.setxor1d "jax.numpy.setxor1d"): the set XOR of two 1D arrays.

- [`jax.numpy.union1d()`](jax.numpy.union1d.html#jax.numpy.union1d "jax.numpy.union1d"): the set union of two 1D arrays.

Examples

Computing the set difference of two arrays:

    >>> ar1 = jnp.array([1, 2, 3, 4])
    >>> ar2 = jnp.array([3, 4, 5, 6])
    >>> jnp.setdiff1d(ar1, ar2)
    Array([1, 2], dtype=int32)

Because the output shape is dynamic, this will fail under [`jit()`](jax.jit.html#jax.jit "jax.jit") and other transformations:

    >>> jax.jit(jnp.setdiff1d)(ar1, ar2)  
    Traceback (most recent call last):
       ...
    ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape int32[4].
    The error occurred while tracing the function setdiff1d at /Users/vanderplas/github/jax-ml/jax/jax/_src/numpy/setops.py:64 for jit. This concrete value was not available in Python because it depends on the value of the argument ar1.

In order to ensure statically-known output shapes, you can pass a static `size` argument:

    >>> jit_setdiff1d = jax.jit(jnp.setdiff1d, static_argnames=['size'])
    >>> jit_setdiff1d(ar1, ar2, size=2)
    Array([1, 2], dtype=int32)

If `size` is too small, the difference is truncated:

    >>> jit_setdiff1d(ar1, ar2, size=1)
    Array([1], dtype=int32)

If `size` is too large, then the output is padded with `fill_value`:

    >>> jit_setdiff1d(ar1, ar2, size=4, fill_value=0)
    Array([1, 2, 0, 0], dtype=int32)

[](jax.numpy.set_printoptions.html "previous page")

previous

jax.numpy.set_printoptions

[](jax.numpy.setxor1d.html "next page")

next

jax.numpy.setxor1d

Contents

- [`setdiff1d()`](#jax.numpy.setdiff1d)

By The JAX authors

© Copyright 2024, The JAX Authors.\
