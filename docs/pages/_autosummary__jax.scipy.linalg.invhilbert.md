- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.invhilbert

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.invhilbert.rst "Download source file")
-  .pdf

# jax.scipy.linalg.invhilbert

## Contents

- [`invhilbert()`](#jax.scipy.linalg.invhilbert)

# jax.scipy.linalg.invhilbert[\#](#jax-scipy-linalg-invhilbert "Link to this heading")

jax.scipy.linalg.invhilbert(*n*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2916-L2967)[\#](#jax.scipy.linalg.invhilbert "Link to this definition")  
Compute the inverse of the Hilbert matrix of order n.

JAX implementation of [`scipy.linalg.invhilbert()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.invhilbert.html#scipy.linalg.invhilbert "(in SciPy v1.19.0.dev)").

The entries are given by a closed-form expression in terms of binomial coefficients:

\\H^{-1}\_{ij} = (-1)^{i + j} (i + j + 1) \binom{n + i}{n - j - 1} \binom{n + j}{n - i - 1} \binom{i + j}{i}^2\\

for \\0 \le i, j \< n\\.

Parameters:  
**n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the size of the matrix to create.

Returns:  
The inverse of the Hilbert matrix of shape `(n,`` ``n)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Unlike [`scipy.linalg.invhilbert()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.invhilbert.html#scipy.linalg.invhilbert "(in SciPy v1.19.0.dev)"), this function does not support `exact=True`. The result is computed in floating point using [`comb()`](jax.scipy.special.comb.html#jax.scipy.special.comb "jax.scipy.special.comb"); for large `n` the entries quickly exceed the dynamic range of finite-precision floats.

See also

[`jax.scipy.linalg.hilbert()`](jax.scipy.linalg.hilbert.html#jax.scipy.linalg.hilbert "jax.scipy.linalg.hilbert")

Examples

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jax.scipy.linalg.invhilbert(2))
    ...   print(jax.scipy.linalg.invhilbert(3))
    [[ 4. -6.]
     [-6. 12.]]
    [[   9.  -36.   30.]
     [ -36.  192. -180.]
     [  30. -180.  180.]]

[](jax.scipy.linalg.inv.html "previous page")

previous

jax.scipy.linalg.inv

[](jax.scipy.linalg.invpascal.html "next page")

next

jax.scipy.linalg.invpascal

Contents

- [`invhilbert()`](#jax.scipy.linalg.invhilbert)

By The JAX authors

© Copyright 2024, The JAX Authors.\
