- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.unwrap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.unwrap.rst "Download source file")
-  .pdf

# jax.numpy.unwrap

## Contents

- [`unwrap()`](#jax.numpy.unwrap)

# jax.numpy.unwrap[\#](#jax-numpy-unwrap "Link to this heading")

jax.numpy.unwrap(*p*, *discont=None*, *axis=-1*, *period=6.283185307179586*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3765-L3855)[\#](#jax.numpy.unwrap "Link to this definition")  
Unwrap a periodic signal.

JAX implementation of [`numpy.unwrap()`](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html#numpy.unwrap "(in NumPy v2.4)").

Parameters:  
- **p** (*ArrayLike*) – input array

- **discont** (*ArrayLike* *\|* *None*) – the maximum allowable discontinuity in the sequence. The default is `period`` ``/`` ``2`

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to unwrap; defaults to -1

- **period** (*ArrayLike*) – the period of the signal, which defaults to \\2\pi\\

Returns:  
An unwrapped copy of `p`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Notes

This implementation follows that of [`numpy.unwrap()`](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html#numpy.unwrap "(in NumPy v2.4)"), and is not well-suited for integer-period unwrapping of narrow-width integers (e.g. int8, int16) or unsigned integers.

Examples

Consider a situation in which you are making measurements of the position of a rotating disk via the `x` and `y` locations of some point on that disk. The underlying variable is an always-increasing angle which we’ll generate this way, using degrees for ease of representation:

    >>> rng = np.random.default_rng(0)
    >>> theta = rng.integers(0, 90, size=(20,)).cumsum()
    >>> theta
    array([ 76, 133, 179, 203, 230, 233, 239, 240, 255, 328, 386, 468, 513,
           567, 654, 719, 775, 823, 873, 957])

Our observations of this angle are the `x` and `y` coordinates, given by the sine and cosine of this underlying angle:

    >>> x, y = jnp.sin(jnp.deg2rad(theta)), jnp.cos(jnp.deg2rad(theta))

Now, say that given these `x` and `y` coordinates, we wish to recover the original angle `theta`. We might do this via the [`atan2()`](jax.numpy.atan2.html#jax.numpy.atan2 "jax.numpy.atan2") function:

    >>> theta_out = jnp.rad2deg(jnp.atan2(x, y)).round()
    >>> theta_out
    Array([  76.,  133.,  179., -157., -130., -127., -121., -120., -105.,
            -32.,   26.,  108.,  153., -153.,  -66.,   -1.,   55.,  103.,
            153., -123.], dtype=float32)

The first few values match the input angle `theta` above, but after this the values are wrapped because the `sin` and `cos` observations obscure the phase information. The purpose of the [`unwrap()`](#jax.numpy.unwrap "jax.numpy.unwrap") function is to recover the original signal from this wrapped view of it:

    >>> jnp.unwrap(theta_out, period=360)
    Array([ 76., 133., 179., 203., 230., 233., 239., 240., 255., 328., 386.,
           468., 513., 567., 654., 719., 775., 823., 873., 957.],      dtype=float32)

It does this by assuming that the true underlying sequence does not differ by more than `discont` (which defaults to `period`` ``/`` ``2`) within a single step, and when it encounters a larger discontinuity it adds factors of the period to the data. For periodic signals that satisfy this assumption, [`unwrap()`](#jax.numpy.unwrap "jax.numpy.unwrap") can recover the original phased signal.

[](jax.numpy.unsignedinteger.html "previous page")

previous

jax.numpy.unsignedinteger

[](jax.numpy.vander.html "next page")

next

jax.numpy.vander

Contents

- [`unwrap()`](#jax.numpy.unwrap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
