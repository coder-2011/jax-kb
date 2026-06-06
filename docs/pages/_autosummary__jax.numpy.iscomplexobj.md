- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.iscomplexobj

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.iscomplexobj.rst "Download source file")
-  .pdf

# jax.numpy.iscomplexobj

## Contents

- [`iscomplexobj()`](#jax.numpy.iscomplexobj)

# jax.numpy.iscomplexobj[\#](#jax-numpy-iscomplexobj "Link to this heading")

jax.numpy.iscomplexobj(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L113-L156)[\#](#jax.numpy.iscomplexobj "Link to this definition")  
Check if the input is a complex number or an array containing complex elements.

JAX implementation of [`numpy.iscomplexobj()`](https://numpy.org/doc/stable/reference/generated/numpy.iscomplexobj.html#numpy.iscomplexobj "(in NumPy v2.4)").

The function evaluates based on input type rather than value. Inputs with zero imaginary parts are still considered complex.

Parameters:  
**x** (*Any*) – input object to check.

Returns:  
True if `x` is a complex number or an array containing at least one complex element, False otherwise.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

See also

- [`jax.numpy.isrealobj()`](jax.numpy.isrealobj.html#jax.numpy.isrealobj "jax.numpy.isrealobj")

- [`jax.numpy.iscomplex()`](jax.numpy.iscomplex.html#jax.numpy.iscomplex "jax.numpy.iscomplex")

Examples

    >>> jnp.iscomplexobj(True)
    False
    >>> jnp.iscomplexobj(0)
    False
    >>> jnp.iscomplexobj(jnp.array([1, 2]))
    False
    >>> jnp.iscomplexobj(1+2j)
    True
    >>> jnp.iscomplexobj(jnp.array([0, 1+2j]))
    True

[](jax.numpy.iscomplex.html "previous page")

previous

jax.numpy.iscomplex

[](jax.numpy.isdtype.html "next page")

next

jax.numpy.isdtype

Contents

- [`iscomplexobj()`](#jax.numpy.iscomplexobj)

By The JAX authors

© Copyright 2024, The JAX Authors.\
