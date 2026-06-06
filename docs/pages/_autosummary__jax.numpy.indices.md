- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.indices

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.indices.rst "Download source file")
-  .pdf

# jax.numpy.indices

## Contents

- [`indices()`](#jax.numpy.indices)

# jax.numpy.indices[\#](#jax-numpy-indices "Link to this heading")

jax.numpy.indices(*dimensions*, *dtype=None*, *sparse=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6208-L6257)[\#](#jax.numpy.indices "Link to this definition")  
Generate arrays of grid indices.

JAX implementation of [`numpy.indices()`](https://numpy.org/doc/stable/reference/generated/numpy.indices.html#numpy.indices "(in NumPy v2.4)").

Parameters:  
- **dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – the shape of the grid.

- **dtype** (*DTypeLike* *\|* *None*) – the dtype of the indices (defaults to integer).

- **sparse** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then return sparse indices. Default is False, which returns dense indices.

Returns:  
An array of shape `(len(dimensions),`` ``*dimensions)` If `sparse` is False, or a sequence of arrays of the same length as `dimensions` if `sparse` is True.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), …\]

See also

- [`jax.numpy.meshgrid()`](jax.numpy.meshgrid.html#jax.numpy.meshgrid "jax.numpy.meshgrid"): generate a grid from arbitrary input arrays.

- [`jax.numpy.mgrid`](jax.numpy.mgrid.html#jax.numpy.mgrid "jax.numpy.mgrid"): generate dense indices using a slicing syntax.

- [`jax.numpy.ogrid`](jax.numpy.ogrid.html#jax.numpy.ogrid "jax.numpy.ogrid"): generate sparse indices using a slicing syntax.

Examples

    >>> jnp.indices((2, 3))
    Array([[[0, 0, 0],
            [1, 1, 1]],

           [[0, 1, 2],
            [0, 1, 2]]], dtype=int32)
    >>> jnp.indices((2, 3), sparse=True)
    (Array([[0],
           [1]], dtype=int32), Array([[0, 1, 2]], dtype=int32))

[](jax.numpy.index_exp.html "previous page")

previous

jax.numpy.index_exp

[](jax.numpy.inexact.html "next page")

next

jax.numpy.inexact

Contents

- [`indices()`](#jax.numpy.indices)

By The JAX authors

© Copyright 2024, The JAX Authors.\
