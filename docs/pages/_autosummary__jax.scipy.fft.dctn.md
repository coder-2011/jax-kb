- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.fft.dctn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.fft.dctn.rst "Download source file")
-  .pdf

# jax.scipy.fft.dctn

## Contents

- [`dctn()`](#jax.scipy.fft.dctn)

# jax.scipy.fft.dctn[\#](#jax-scipy-fft-dctn "Link to this heading")

jax.scipy.fft.dctn(*x*, *type=2*, *s=None*, *axes=None*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/fft.py#L152-L264)[\#](#jax.scipy.fft.dctn "Link to this definition")  
Computes the multidimensional discrete cosine transform of the input

JAX implementation of [`scipy.fft.dctn()`](https://scipy.github.io/devdocs/reference/generated/scipy.fft.dctn.html#scipy.fft.dctn "(in SciPy v1.19.0.dev)").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array

- **type** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default = 2. Currently only type 2 is supported.

- **s** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – integer or sequence of integers. Specifies the shape of the result. If not specified, it will default to the shape of `x` along the specified `axes`.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – integer or sequence of integers. Specifies the axes along which the transform will be computed. If not given, the last `len(s)` axes are used, or all axes if `s` is also not specified.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode: one of `[None,`` ``"backward",`` ``"ortho"]`. The default is `None`, which is equivalent to `"backward"`.

Returns:  
array containing the discrete cosine transform of x

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.fft.dct()`](jax.scipy.fft.dct.html#jax.scipy.fft.dct "jax.scipy.fft.dct"): one-dimensional DCT

- [`jax.scipy.fft.idct()`](jax.scipy.fft.idct.html#jax.scipy.fft.idct "jax.scipy.fft.idct"): one-dimensional inverse DCT

- [`jax.scipy.fft.idctn()`](jax.scipy.fft.idctn.html#jax.scipy.fft.idctn "jax.scipy.fft.idctn"): multidimensional inverse DCT

Examples

`jax.scipy.fft.dctn` computes the transform along both the axes by default when `axes` argument is `None` and `s` is also `None`.

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x))
    [[ 12.01   6.2  -10.17]
     [  8.84   9.65  -3.54]
     [ 11.25  -1.54  -0.88]]

When `s=[2]`, the transform will be computed only along the last axis, with its dimension padded or truncated to size `2`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2]))
    [[ 7.3  -0.57]
     [ 0.19 -0.36]
     [-0.   -1.4 ]]

When `s=[2]` and `axes=[0]`, the transform will be computed only along the specified axis, with its dimension padded or truncated to size `2`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2], axes=[0]))
    [[ 3.09  4.4  -2.81]
     [ 2.41  2.62  0.76]]

When `s=[2,`` ``4]`, shape of the transform will be `(2,`` ``4)`.

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dctn(x, s=[2, 4]))
    [[  9.36  11.23   2.12 -10.97]
     [ 11.57   5.86  -1.37  -1.58]]

[](jax.scipy.fft.dct.html "previous page")

previous

jax.scipy.fft.dct

[](jax.scipy.fft.idct.html "next page")

next

jax.scipy.fft.idct

Contents

- [`dctn()`](#jax.scipy.fft.dctn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
