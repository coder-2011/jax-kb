- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.fft.dct

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.fft.dct.rst "Download source file")
-  .pdf

# jax.scipy.fft.dct

## Contents

- [`dct()`](#jax.scipy.fft.dct)

# jax.scipy.fft.dct[\#](#jax-scipy-fft-dct "Link to this heading")

jax.scipy.fft.dct(*x*, *type=2*, *n=None*, *axis=-1*, *norm=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/fft.py#L51-L134)[\#](#jax.scipy.fft.dct "Link to this definition")  
Computes the discrete cosine transform of the input

JAX implementation of [`scipy.fft.dct()`](https://scipy.github.io/devdocs/reference/generated/scipy.fft.dct.html#scipy.fft.dct "(in SciPy v1.19.0.dev)").

Parameters:  
- **x** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – array

- **type** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default = 2. Currently only type 2 is supported.

- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – integer, default = x.shape\[axis\]. The length of the transform. If larger than `x.shape[axis]`, the input will be zero-padded, if smaller, the input will be truncated.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer, default=-1. The axis along which the dct will be performed.

- **norm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string. The normalization mode: one of `[None,`` ``"backward",`` ``"ortho"]`. The default is `None`, which is equivalent to `"backward"`.

Returns:  
array containing the discrete cosine transform of x

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.scipy.fft.dctn()`](jax.scipy.fft.dctn.html#jax.scipy.fft.dctn "jax.scipy.fft.dctn"): multidimensional DCT

- [`jax.scipy.fft.idct()`](jax.scipy.fft.idct.html#jax.scipy.fft.idct "jax.scipy.fft.idct"): inverse DCT

- [`jax.scipy.fft.idctn()`](jax.scipy.fft.idctn.html#jax.scipy.fft.idctn "jax.scipy.fft.idctn"): multidimensional inverse DCT

Examples

    >>> x = jax.random.normal(jax.random.key(0), (3, 3))
    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x))
    [[ 6.43  3.56 -2.86]
     [-1.75  1.55 -1.4 ]
     [ 1.33 -2.01 -0.82]]

When `n` smaller than `x.shape[axis]`

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=2))
    [[ 7.3  -0.57]
     [ 0.19 -0.36]
     [-0.   -1.4 ]]

When `n` smaller than `x.shape[axis]` and `axis=0`

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=2, axis=0))
    [[ 3.09  4.4  -2.81]
     [ 2.41  2.62  0.76]]

When `n` larger than `x.shape[axis]` and `axis=1`

    >>> with jnp.printoptions(precision=2, suppress=True):
    ...   print(jax.scipy.fft.dct(x, n=4, axis=1))
    [[ 6.43  4.88  0.04 -3.3 ]
     [-1.75  0.73  1.01 -2.18]
     [ 1.33 -1.05 -2.34 -0.07]]

[](jax.scipy.cluster.vq.vq.html "previous page")

previous

jax.scipy.cluster.vq.vq

[](jax.scipy.fft.dctn.html "next page")

next

jax.scipy.fft.dctn

Contents

- [`dct()`](#jax.scipy.fft.dct)

By The JAX authors

© Copyright 2024, The JAX Authors.\
