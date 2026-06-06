- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.fft.idct

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.fft.idct.rst "Download source file")
-  .pdf

# jax.scipy.fft.idct

## Contents

- [`idct()`](#jax.scipy.fft.idct)

# jax.scipy.fft.idct[\#](#jax-scipy-fft-idct "Link to this heading")

jax.scipy.fft.idct(*x*, *type=2*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/fft.py#L266-L365)[\#](#jax.scipy.fft.idct "Link to this definition")  
Computes the inverse discrete cosine transform of the input

JAX implementation of [`scipy.fft.idct()`](https://scipy.github.io/devdocs/reference/generated/scipy.fft.idct.html#scipy.fft.idct "(in SciPy v1.19.0.dev)").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array

- **type** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default = 2. Currently only type 2 is supported.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer, default = x.shape\[axis\]. The length of the transform. If larger than `x.shape[axis]`, the input will be zero-padded, if smaller, the input will be truncated.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default=-1. The axis along which the dct will be performed.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode: one of `[None,`` ``"backward",`` ``"ortho"]`. The default is `None`, which is equivalent to `"backward"`.

Returns:  
array containing the inverse discrete cosine transform of x

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.fft.dct()`](jax.scipy.fft.dct.html#jax.scipy.fft.dct "jax.scipy.fft.dct"): DCT

- [`jax.scipy.fft.dctn()`](jax.scipy.fft.dctn.html#jax.scipy.fft.dctn "jax.scipy.fft.dctn"): multidimensional DCT

- [`jax.scipy.fft.idctn()`](jax.scipy.fft.idctn.html#jax.scipy.fft.idctn "jax.scipy.fft.idctn"): multidimensional inverse DCT

Examples

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x))
    [[ 0.78  0.41 -0.39]
     [-0.12  0.31 -0.23]
     [ 0.17 -0.3  -0.11]]

When `n` smaller than `x.shape[axis]`

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=2))
    [[ 1.12 -0.31]
     [ 0.04 -0.08]
     [ 0.05 -0.3 ]]

When `n` smaller than `x.shape[axis]` and `axis=0`

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=2, axis=0))
    [[ 0.38  0.57 -0.45]
     [ 0.43  0.44  0.24]]

When `n` larger than `x.shape[axis]` and `axis=0`

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...    print(jax.scipy.fft.idct(x, n=4, axis=0))
    [[ 0.1   0.38 -0.16]
     [ 0.28  0.18 -0.26]
     [ 0.3   0.15 -0.08]
     [ 0.13  0.3   0.29]]

`jax.scipy.fft.idct` can be used to reconstruct `x` from the result of `jax.scipy.fft.dct`

    >>> x_dct = jax.scipy.fft.dct(x)
    >>> jnp.allclose(x, jax.scipy.fft.idct(x_dct))
    Array(True, dtype=bool)

[](jax.scipy.fft.dctn.html "previous page")

previous

jax.scipy.fft.dctn

[](jax.scipy.fft.idctn.html "next page")

next

jax.scipy.fft.idctn

Contents

- [`idct()`](#jax.scipy.fft.idct)

By The JAX authors

© Copyright 2024, The JAX Authors.\
