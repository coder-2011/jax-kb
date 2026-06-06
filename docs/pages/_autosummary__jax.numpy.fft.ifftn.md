- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fft.ifftn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fft.ifftn.rst "Download source file")
-  .pdf

# jax.numpy.fft.ifftn

## Contents

- [`ifftn()`](#jax.numpy.fft.ifftn)

# jax.numpy.fft.ifftn[\#](#jax-numpy-fft-ifftn "Link to this heading")

jax.numpy.fft.ifftn(*a*, *s=None*, *axes=None*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/fft.py#L206-L272)[\#](#jax.numpy.fft.ifftn "Link to this definition")  
Compute a multidimensional inverse discrete Fourier transform.

JAX implementation of [`numpy.fft.ifftn()`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftn.html#numpy.fft.ifftn "(in NumPy v2.4)").

Parameters:  
- **a** (*ArrayLike*) – input array

- **s** (*Shape* *\|* *None*) – sequence of integers. Specifies the shape of the result. If not specified, it will default to the shape of `a` along the specified `axes`.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – sequence of integers, default=None. Specifies the axes along which the transform is computed. If None, computes the transform along all the axes.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode. “backward”, “ortho” and “forward” are supported.

Returns:  
An array containing the multidimensional inverse discrete Fourier transform of `a`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.fft.fftn()`](jax.numpy.fft.fftn.html#jax.numpy.fft.fftn "jax.numpy.fft.fftn"): Computes a multidimensional discrete Fourier transform.

- [`jax.numpy.fft.fft()`](jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft"): Computes a one-dimensional discrete Fourier transform.

- [`jax.numpy.fft.ifft()`](jax.numpy.fft.ifft.html#jax.numpy.fft.ifft "jax.numpy.fft.ifft"): Computes a one-dimensional inverse discrete Fourier transform.

Examples

`jnp.fft.ifftn` computes the transform along all the axes by default when `axes` argument is `None`.

    >>> x = jnp.array([[1, 2, 5, 3],
    ...                [4, 1, 2, 6],
    ...                [5, 3, 2, 1]])
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x))
    [[ 2.92+0.j    0.08-0.33j  0.25+0.j    0.08+0.33j]
     [-0.08+0.14j -0.04-0.03j  0.  -0.29j -1.05-0.11j]
     [-0.08-0.14j -1.05+0.11j  0.  +0.29j -0.04+0.03j]]

When `s=[3]`, dimension of the transform along `axis`` ``-1` will be `3` and dimension along other axes will be the same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x, s=[3]))
    [[ 2.67+0.j   -0.83-0.87j -0.83+0.87j]
     [ 2.33+0.j    0.83-0.29j  0.83+0.29j]
     [ 3.33+0.j    0.83+0.29j  0.83-0.29j]]

When `s=[2]` and `axes=[0]`, dimension of the transform along `axis`` ``0` will be `2` and dimension along other axes will be same as that of input.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x, s=[2], axes=[0]))
    [[ 2.5+0.j  1.5+0.j  3.5+0.j  4.5+0.j]
     [-1.5+0.j  0.5+0.j  1.5+0.j -1.5+0.j]]

When `s=[2,`` ``3]`, shape of the transform will be `(2,`` ``3)`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jnp.fft.ifftn(x, s=[2, 3]))
    [[ 2.5 +0.j    0.  -0.58j  0.  +0.58j]
     [ 0.17+0.j   -0.83-0.29j -0.83+0.29j]]

[](jax.numpy.fft.ifft2.html "previous page")

previous

jax.numpy.fft.ifft2

[](jax.numpy.fft.ifftshift.html "next page")

next

jax.numpy.fft.ifftshift

Contents

- [`ifftn()`](#jax.numpy.fft.ifftn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
