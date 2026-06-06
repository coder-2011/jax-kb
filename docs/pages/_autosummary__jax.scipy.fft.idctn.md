- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.fft.idctn

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.fft.idctn.rst "Download source file")
-  .pdf

# jax.scipy.fft.idctn

## Contents

- [`idctn()`](#jax.scipy.fft.idctn)

# jax.scipy.fft.idctn[\#](#jax-scipy-fft-idctn "Link to this heading")

jax.scipy.fft.idctn(*x*, *type=2*, *s=None*, *axes=None*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/fft.py#L367-L483)[\#](#jax.scipy.fft.idctn "Link to this definition")  
Computes the multidimensional inverse discrete cosine transform of the input

JAX implementation of [`scipy.fft.idctn()`](https://scipy.github.io/devdocs/reference/generated/scipy.fft.idctn.html#scipy.fft.idctn "(in SciPy v1.19.0.dev)").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array

- **type** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default = 2. Currently only type 2 is supported.

- **s** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – integer or sequence of integers. Specifies the shape of the result. If not specified, it will default to the shape of `x` along the specified `axes`.

- **axes** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – integer or sequence of integers. Specifies the axes along which the transform will be computed. If not given, the last `len(s)` axes are used, or all axes if `s` is also not specified.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode: one of `[None,`` ``"backward",`` ``"ortho"]`. The default is `None`, which is equivalent to `"backward"`.

Returns:  
array containing the inverse discrete cosine transform of x

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.fft.dct()`](jax.scipy.fft.dct.html#jax.scipy.fft.dct "jax.scipy.fft.dct"): one-dimensional DCT

- [`jax.scipy.fft.dctn()`](jax.scipy.fft.dctn.html#jax.scipy.fft.dctn "jax.scipy.fft.dctn"): multidimensional DCT

- [`jax.scipy.fft.idct()`](jax.scipy.fft.idct.html#jax.scipy.fft.idct "jax.scipy.fft.idct"): one-dimensional inverse DCT

Examples

`jax.scipy.fft.idctn` computes the transform along both the axes by default when `axes` argument is `None` and `s` is also `None`.

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idctn(x))
    [[ 0.12  0.11 -0.15]
     [ 0.07  0.17 -0.03]
     [ 0.19 -0.07 -0.02]]

When `s=[2]`, the transform will be computed only along the last axis, with its dimension padded or truncated to size `2`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2]))
    [[ 1.12 -0.31]
     [ 0.04 -0.08]
     [ 0.05 -0.3 ]]

When `s=[2]` and `axes=[0]`, the transform will be computed only along the specified axis, with its dimension padded or truncated to size `2`:

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2], axes=[0]))
    [[ 0.38  0.57 -0.45]
     [ 0.43  0.44  0.24]]

When `s=[2,`` ``4]`, shape of the transform will be `(2,`` ``4)`

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...  print(jax.scipy.fft.idctn(x, s=[2, 4]))
    [[ 0.1   0.18  0.07 -0.16]
     [ 0.2   0.06 -0.03 -0.01]]

`jax.scipy.fft.idctn` can be used to reconstruct `x` from the result of `jax.scipy.fft.dctn`

    >>> x_dctn = jax.scipy.fft.dctn(x)
    >>> jnp.allclose(x, jax.scipy.fft.idctn(x_dctn))
    Array(True, dtype=bool)

[](jax.scipy.fft.idct.html "previous page")

previous

jax.scipy.fft.idct

[](jax.scipy.integrate.trapezoid.html "next page")

next

jax.scipy.integrate.trapezoid

Contents

- [`idctn()`](#jax.scipy.fft.idctn)

By The JAX authors

© Copyright 2024, The JAX Authors.\
