- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.left_shift

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.left_shift.rst "Download source file")
-  .pdf

# jax.numpy.left_shift

## Contents

- [`left_shift()`](#jax.numpy.left_shift)

# jax.numpy.left_shift[\#](#jax-numpy-left-shift "Link to this heading")

jax.numpy.left_shift(*x*, *y*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufuncs.py#L1374-L1428)[\#](#jax.numpy.left_shift "Link to this definition")  
Shift bits of `x` to left by the amount specified in `y`, element-wise.

JAX implementation of [`numpy.left_shift`](https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html#numpy.left_shift "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – Input array, must be integer-typed.

- **y** (*ArrayLike*) – The amount of bits to shift each element in `x` to the left, only accepts integer subtypes. `x` and `y` must either have same shape or be broadcast compatible.

Returns:  
An array containing the left shifted elements of `x` by the amount specified in `y`, with the same shape as the broadcasted shape of `x` and `y`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Left shifting `x` by `y` is equivalent to `x`` ``*`` ``(2**y)` within the bounds of the dtypes involved.

See also

- [`jax.numpy.right_shift()`](jax.numpy.right_shift.html#jax.numpy.right_shift "jax.numpy.right_shift"): and [`jax.numpy.bitwise_right_shift()`](jax.numpy.bitwise_right_shift.html#jax.numpy.bitwise_right_shift "jax.numpy.bitwise_right_shift"): Shifts the bits of `x1` to right by the amount specified in `x2`, element-wise.

- [`jax.numpy.bitwise_left_shift()`](jax.numpy.bitwise_left_shift.html#jax.numpy.bitwise_left_shift "jax.numpy.bitwise_left_shift"): Alias of `jax.left_shift()`.

Examples

    >>> def print_binary(x):
    ...   return [bin(int(val)) for val in x]

    >>> x1 = jnp.arange(5)
    >>> x1
    Array([0, 1, 2, 3, 4], dtype=int32)
    >>> print_binary(x1)
    ['0b0', '0b1', '0b10', '0b11', '0b100']
    >>> x2 = 1
    >>> result = jnp.left_shift(x1, x2)
    >>> result
    Array([0, 2, 4, 6, 8], dtype=int32)
    >>> print_binary(result)
    ['0b0', '0b10', '0b100', '0b110', '0b1000']

    >>> x3 = 4
    >>> print_binary([x3])
    ['0b100']
    >>> x4 = jnp.array([1, 2, 3, 4])
    >>> result1 = jnp.left_shift(x3, x4)
    >>> result1
    Array([ 8, 16, 32, 64], dtype=int32)
    >>> print_binary(result1)
    ['0b1000', '0b10000', '0b100000', '0b1000000']

[](jax.numpy.ldexp.html "previous page")

previous

jax.numpy.ldexp

[](jax.numpy.less.html "next page")

next

jax.numpy.less

Contents

- [`left_shift()`](#jax.numpy.left_shift)

By The JAX authors

© Copyright 2024, The JAX Authors.\
