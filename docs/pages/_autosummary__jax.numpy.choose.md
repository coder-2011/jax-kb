- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.choose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.choose.rst "Download source file")
-  .pdf

# jax.numpy.choose

## Contents

- [`choose()`](#jax.numpy.choose)

# jax.numpy.choose[\#](#jax-numpy-choose "Link to this heading")

jax.numpy.choose(*a*, *choices*, *out=None*, *mode='raise'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L4863-L4965)[\#](#jax.numpy.choose "Link to this definition")  
Construct an array by stacking slices of choice arrays.

JAX implementation of [`numpy.choose()`](https://numpy.org/doc/stable/reference/generated/numpy.choose.html#numpy.choose "(in NumPy v2.4)").

The semantics of this function can be confusing, but in the simplest case where `a` is a one-dimensional array, `choices` is a two-dimensional array, and all entries of `a` are in-bounds (i.e. `0`` ``<=`` ``a_i`` ``<`` ``len(choices)`), then the function is equivalent to the following:

    def choose(a, choices):
      return jnp.array([choices[a_i, i] for i, a_i in enumerate(a)])

In the more general case, `a` may have any number of dimensions and `choices` may be an arbitrary sequence of broadcast-compatible arrays. In this case, again for in-bound indices, the logic is equivalent to:

    def choose(a, choices):
      a, *choices = jnp.broadcast_arrays(a, *choices)
      choices = jnp.array(choices)
      return jnp.array([choices[a[idx], *idx] for idx in np.ndindex(a.shape)])

The only additional complexity comes from the `mode` argument, which controls the behavior for out-of-bound indices in `a` as described below.

Parameters:  
- **a** (*ArrayLike*) – an N-dimensional array of integer indices.

- **choices** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *np.ndarray* *\|* *Sequence\[ArrayLike\]*) – an array or sequence of arrays. All arrays in the sequence must be mutually broadcast compatible with `a`.

- **out** (*None*) – unused by JAX

- **mode** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – specify the out-of-bounds indexing mode; one of `'raise'` (default), `'wrap'`, or `'clip'`. Note that the default mode of `'raise'` is not compatible with JAX transformations.

Returns:  
an array containing stacked slices from `choices` at the indices specified by `a`. The shape of the result is `broadcast_shapes(a.shape,`` ``*(c.shape`` ``for`` ``c`` ``in`` ``choices))`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.switch()`](jax.lax.switch.html#jax.lax.switch "jax.lax.switch"): choose between N functions based on an index.

Examples

Here is the simplest case of a 1D index array with a 2D choice array, in which case this chooses the indexed value from each column:

    >>> choices = jnp.array([[ 1,  2,  3,  4],
    ...                      [ 5,  6,  7,  8],
    ...                      [ 9, 10, 11, 12]])
    >>> a = jnp.array([2, 0, 1, 0])
    >>> jnp.choose(a, choices)
    Array([9, 2, 7, 4], dtype=int32)

The `mode` argument specifies what to do with out-of-bound indices; options are to either `wrap` or `clip`:

    >>> a2 = jnp.array([2, 0, 1, 4])  # last index out-of-bound
    >>> jnp.choose(a2, choices, mode='clip')
    Array([ 9,  2,  7, 12], dtype=int32)
    >>> jnp.choose(a2, choices, mode='wrap')
    Array([9, 2, 7, 8], dtype=int32)

In the more general case, `choices` may be a sequence of array-like objects with any broadcast-compatible shapes.

    >>> choice_1 = jnp.array([1, 2, 3, 4])
    >>> choice_2 = 99
    >>> choice_3 = jnp.array([[10],
    ...                       [20],
    ...                       [30]])
    >>> a = jnp.array([[0, 1, 2, 0],
    ...                [1, 2, 0, 1],
    ...                [2, 0, 1, 2]])
    >>> jnp.choose(a, [choice_1, choice_2, choice_3], mode='wrap')
    Array([[ 1, 99, 10,  4],
           [99, 20,  3, 99],
           [30,  2, 99, 30]], dtype=int32)

[](jax.numpy.character.html "previous page")

previous

jax.numpy.character

[](jax.numpy.clip.html "next page")

next

jax.numpy.clip

Contents

- [`choose()`](#jax.numpy.choose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
