- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.i0

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.i0.rst "Download source file")
-  .pdf

# jax.numpy.i0

## Contents

- [`i0()`](#jax.numpy.i0)

# jax.numpy.i0[\#](#jax-numpy-i0 "Link to this heading")

jax.numpy.i0(*x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L6089-L6127)[\#](#jax.numpy.i0 "Link to this definition")  
Calculate modified Bessel function of first kind, zeroth order.

JAX implementation of [`numpy.i0()`](https://numpy.org/doc/stable/reference/generated/numpy.i0.html#numpy.i0 "(in NumPy v2.4)").

Modified Bessel function of first kind, zeroth order is defined by:

\\\mathrm{i0}(x) = I_0(x) = \sum\_{k=0}^{\infty} \frac{(x^2/4)^k}{(k!)^2}\\

Parameters:  
**x** (*ArrayLike*) – scalar or array. Specifies the argument of Bessel function. Complex inputs are not supported.

Returns:  
An array containing the corresponding values of the modified Bessel function of `x`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.special.i0()`](jax.scipy.special.i0.html#jax.scipy.special.i0 "jax.scipy.special.i0"): Calculates the modified Bessel function of zeroth order.

- [`jax.scipy.special.i1()`](jax.scipy.special.i1.html#jax.scipy.special.i1 "jax.scipy.special.i1"): Calculates the modified Bessel function of first order.

- [`jax.scipy.special.i0e()`](jax.scipy.special.i0e.html#jax.scipy.special.i0e "jax.scipy.special.i0e"): Calculates the exponentially scaled modified Bessel function of zeroth order.

Examples

    >>> x = jnp.array([-2, -1, 0, 1, 2])
    >>> jnp.i0(x)
    Array([2.2795851, 1.266066 , 1.0000001, 1.266066 , 2.2795851], dtype=float32)

[](jax.numpy.hypot.html "previous page")

previous

jax.numpy.hypot

[](jax.numpy.identity.html "next page")

next

jax.numpy.identity

Contents

- [`i0()`](#jax.numpy.i0)

By The JAX authors

© Copyright 2024, The JAX Authors.\
