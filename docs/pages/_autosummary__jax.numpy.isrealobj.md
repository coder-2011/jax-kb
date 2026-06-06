- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isrealobj

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isrealobj.rst "Download source file")
-  .pdf

# jax.numpy.isrealobj

## Contents

- [`isrealobj()`](#jax.numpy.isrealobj)

# jax.numpy.isrealobj[\#](#jax-numpy-isrealobj "Link to this heading")

jax.numpy.isrealobj(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L1866-L1899)[\#](#jax.numpy.isrealobj "Link to this definition")  
Check if the input is not a complex number or an array containing complex elements.

JAX implementation of [`numpy.isrealobj()`](https://numpy.org/doc/stable/reference/generated/numpy.isrealobj.html#numpy.isrealobj "(in NumPy v2.4)").

The function evaluates based on input type rather than value. Inputs with zero imaginary parts are still considered complex.

Parameters:  
**x** (*Any*) – input object to check.

Returns:  
False if `x` is a complex number or an array containing at least one complex element, True otherwise.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

See also

- [`jax.numpy.iscomplexobj()`](jax.numpy.iscomplexobj.html#jax.numpy.iscomplexobj "jax.numpy.iscomplexobj")

- [`jax.numpy.isreal()`](jax.numpy.isreal.html#jax.numpy.isreal "jax.numpy.isreal")

Examples

    >>> jnp.isrealobj(0)
    True
    >>> jnp.isrealobj(1.2)
    True
    >>> jnp.isrealobj(jnp.array([1, 2]))
    True
    >>> jnp.isrealobj(1+2j)
    False
    >>> jnp.isrealobj(jnp.array([0, 1+2j]))
    False

[](jax.numpy.isreal.html "previous page")

previous

jax.numpy.isreal

[](jax.numpy.isscalar.html "next page")

next

jax.numpy.isscalar

Contents

- [`isrealobj()`](#jax.numpy.isrealobj)

By The JAX authors

© Copyright 2024, The JAX Authors.\
