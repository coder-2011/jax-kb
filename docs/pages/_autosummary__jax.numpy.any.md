- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.any

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.any.rst "Download source file")
-  .pdf

# jax.numpy.any

## Contents

- [`any()`](#jax.numpy.any)

# jax.numpy.any[\#](#jax-numpy-any "Link to this heading")

jax.numpy.any(*a*, *axis=None*, *out=None*, *keepdims=False*, *\**, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L628-L677)[\#](#jax.numpy.any "Link to this definition")  
Test whether any of the array elements along a given axis evaluate to True.

JAX implementation of [`numpy.any()`](https://numpy.org/doc/stable/reference/generated/numpy.any.html#numpy.any "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – Input array.

- **axis** (*Axis*) – int or array, default=None. Axis along which to be tested. If None, tests along all the axes.

- **keepdims** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – bool, default=False. If true, reduced axes are left in the result with size 1.

- **where** (*ArrayLike* *\|* *None*) – int or array of boolean dtype, default=None. The elements to be used in the test. Array should be broadcast compatible to the input.

- **out** (*None*) – Unused by JAX.

Returns:  
An array of boolean values.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

By default, `jnp.any` tests along all the axes.

    >>> x = jnp.array([[True, True, True, False],
    ...                [True, False, True, False],
    ...                [True, True, False, False]])
    >>> jnp.any(x)
    Array(True, dtype=bool)

If `axis=0`, tests along axis 0.

    >>> jnp.any(x, axis=0)
    Array([ True,  True,  True, False], dtype=bool)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.any(x, axis=0, keepdims=True)
    Array([[ True,  True,  True, False]], dtype=bool)

To include specific elements in testing for True values, you can use a\`\`where\`\`.

    >>> where=jnp.array([[1, 0, 1, 0],
    ...                  [0, 1, 0, 1],
    ...                  [1, 0, 1, 0]], dtype=bool)
    >>> jnp.any(x, axis=0, keepdims=True, where=where)
    Array([[ True, False,  True, False]], dtype=bool)

[](jax.numpy.angle.html "previous page")

previous

jax.numpy.angle

[](jax.numpy.append.html "next page")

next

jax.numpy.append

Contents

- [`any()`](#jax.numpy.any)

By The JAX authors

© Copyright 2024, The JAX Authors.\
