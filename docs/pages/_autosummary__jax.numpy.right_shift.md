- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.right_shift

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.right_shift.rst "Download source file")
-  .pdf

# jax.numpy.right_shift

## Contents

- [`right_shift()`](#jax.numpy.right_shift)

# jax.numpy.right_shift[\#](#jax-numpy-right-shift "Link to this heading")

jax.numpy.right_shift(*x1*, *x2*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L2255-L2305)[\#](#jax.numpy.right_shift "Link to this definition")  
Right shift the bits of `x1` to the amount specified in `x2`.

JAX implementation of [`numpy.right_shift`](https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html#numpy.right_shift "(in NumPy v2.4)").

Parameters:  
- **x1** (*ArrayLike*) – Input array, only accepts unsigned integer subtypes

- **x2** (*ArrayLike*) – The amount of bits to shift each element in `x1` to the right, only accepts integer subtypes

Returns:  
An array-like object containing the right shifted elements of `x1` by the amount specified in `x2`, with the same shape as the broadcasted shape of `x1` and `x2`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

If `x1.shape`` ``!=`` ``x2.shape`, they must be compatible for broadcasting to a shared shape, this shared shape will also be the shape of the output. Right shifting a scalar x1 by scalar x2 is equivalent to `x1`` ``//`` ``2**x2`.

Examples

    >>> def print_binary(x):
    ...   return [bin(int(val)) for val in x]

    >>> x1 = jnp.array([1, 2, 4, 8])
    >>> print_binary(x1)
    ['0b1', '0b10', '0b100', '0b1000']
    >>> x2 = 1
    >>> result = jnp.right_shift(x1, x2)
    >>> result
    Array([0, 1, 2, 4], dtype=int32)
    >>> print_binary(result)
    ['0b0', '0b1', '0b10', '0b100']

    >>> x1 = 16
    >>> print_binary([x1])
    ['0b10000']
    >>> x2 = jnp.array([1, 2, 3, 4])
    >>> result = jnp.right_shift(x1, x2)
    >>> result
    Array([8, 4, 2, 1], dtype=int32)
    >>> print_binary(result)
    ['0b1000', '0b100', '0b10', '0b1']

[](jax.numpy.result_type.html "previous page")

previous

jax.numpy.result_type

[](jax.numpy.rint.html "next page")

next

jax.numpy.rint

Contents

- [`right_shift()`](#jax.numpy.right_shift)

By The JAX authors

© Copyright 2024, The JAX Authors.\
