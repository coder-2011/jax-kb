- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.all

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.all.rst "Download source file")
-  .pdf

# jax.numpy.all

## Contents

- [`all()`](#jax.numpy.all)

# jax.numpy.all[\#](#jax-numpy-all "Link to this heading")

jax.numpy.all(*a*, *axis=None*, *out=None*, *keepdims=False*, *\**, *where=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/reductions.py#L571-L620)[\#](#jax.numpy.all "Link to this definition")  
Test whether all array elements along a given axis evaluate to True.

JAX implementation of [`numpy.all()`](https://numpy.org/doc/stable/reference/generated/numpy.all.html#numpy.all "(in NumPy v2.4)").

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

By default, `jnp.all` tests for True values along all the axes.

    >>> x = jnp.array([[True, True, True, False],
    ...                [True, False, True, False],
    ...                [True, True, False, False]])
    >>> jnp.all(x)
    Array(False, dtype=bool)

If `axis=0`, tests for True values along axis 0.

    >>> jnp.all(x, axis=0)
    Array([ True, False, False, False], dtype=bool)

If `keepdims=True`, `ndim` of the output will be same of that of the input.

    >>> jnp.all(x, axis=0, keepdims=True)
    Array([[ True, False, False, False]], dtype=bool)

To include specific elements in testing for True values, you can use a\`\`where\`\`.

    >>> where=jnp.array([[1, 0, 1, 0],
    ...                  [0, 0, 1, 1],
    ...                  [1, 1, 1, 0]], dtype=bool)
    >>> jnp.all(x, axis=0, keepdims=True, where=where)
    Array([[ True,  True, False, False]], dtype=bool)

[](jax.numpy.add.html "previous page")

previous

jax.numpy.add

[](jax.numpy.allclose.html "next page")

next

jax.numpy.allclose

Contents

- [`all()`](#jax.numpy.all)

By The JAX authors

© Copyright 2024, The JAX Authors.\
