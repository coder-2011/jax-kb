- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.invpascal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.invpascal.rst "Download source file")
-  .pdf

# jax.scipy.linalg.invpascal

## Contents

- [`invpascal()`](#jax.scipy.linalg.invpascal)

# jax.scipy.linalg.invpascal[\#](#jax-scipy-linalg-invpascal "Link to this heading")

jax.scipy.linalg.invpascal(*n*, *kind=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L3031-L3094)[\#](#jax.scipy.linalg.invpascal "Link to this definition")  
Compute the inverse of the Pascal matrix of order n.

JAX implementation of [`scipy.linalg.invpascal()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.invpascal.html#scipy.linalg.invpascal "(in SciPy v1.19.0.dev)").

The inverses of the lower and upper Pascal matrices have a simple closed form,

\\L^{-1}\_{ij} = (-1)^{i - j} \binom{i}{j}, \qquad U^{-1} = (L^{-1})^T,\\

and the symmetric inverse satisfies \\P_S^{-1} = U^{-1} L^{-1}\\.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the size of the matrix to create.

- **kind** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – (optional) must be one of `lower`, `upper`, or `symmetric` (default).

Returns:  
The inverse of the Pascal matrix of shape `(n,`` ``n)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Note

Unlike [`scipy.linalg.invpascal()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.invpascal.html#scipy.linalg.invpascal "(in SciPy v1.19.0.dev)"), this function does not support `exact=True`. The entries are computed in floating point through [`gammaln()`](jax.scipy.special.gammaln.html#jax.scipy.special.gammaln "jax.scipy.special.gammaln")-based binomial coefficients.

See also

[`jax.scipy.linalg.pascal()`](jax.scipy.linalg.pascal.html#jax.scipy.linalg.pascal "jax.scipy.linalg.pascal")

Examples

    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(jax.scipy.linalg.invpascal(4, kind="lower"))
    ...   print(jax.scipy.linalg.invpascal(5))
    [[ 1. -0.  0. -0.]
     [-1.  1. -0.  0.]
     [ 1. -2.  1. -0.]
     [-1.  3. -3.  1.]]
    [[  5. -10.  10.  -5.   1.]
     [-10.  30. -35.  19.  -4.]
     [ 10. -35.  46. -27.   6.]
     [ -5.  19. -27.  17.  -4.]
     [  1.  -4.   6.  -4.   1.]]

[](jax.scipy.linalg.invhilbert.html "previous page")

previous

jax.scipy.linalg.invhilbert

[](jax.scipy.linalg.leslie.html "next page")

next

jax.scipy.linalg.leslie

Contents

- [`invpascal()`](#jax.scipy.linalg.invpascal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
