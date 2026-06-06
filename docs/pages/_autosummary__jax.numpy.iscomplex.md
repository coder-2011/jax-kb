- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.iscomplex

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.iscomplex.rst "Download source file")
-  .pdf

# jax.numpy.iscomplex

## Contents

- [`iscomplex()`](#jax.numpy.iscomplex)

# jax.numpy.iscomplex[\#](#jax-numpy-iscomplex "Link to this heading")

jax.numpy.iscomplex(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1457-L1480)[\#](#jax.numpy.iscomplex "Link to this definition")  
Return boolean array showing where the input is complex.

JAX implementation of [`numpy.iscomplex()`](https://numpy.org/doc/stable/reference/generated/numpy.iscomplex.html#numpy.iscomplex "(in NumPy v2.4)").

Parameters:  
**x** (*ArrayLike*) – Input array to check.

Returns:  
A new array containing boolean values indicating complex elements.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.iscomplexobj()`](jax.numpy.iscomplexobj.html#jax.numpy.iscomplexobj "jax.numpy.iscomplexobj")

- [`jax.numpy.isrealobj()`](jax.numpy.isrealobj.html#jax.numpy.isrealobj "jax.numpy.isrealobj")

Examples

    >>> jnp.iscomplex(jnp.array([True, 0, 1, 2j, 1+2j]))
    Array([False, False, False, True, True], dtype=bool)

[](jax.numpy.isclose.html "previous page")

previous

jax.numpy.isclose

[](jax.numpy.iscomplexobj.html "next page")

next

jax.numpy.iscomplexobj

Contents

- [`iscomplex()`](#jax.numpy.iscomplex)

By The JAX authors

© Copyright 2024, The JAX Authors.\
