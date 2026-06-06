- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.pascal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.pascal.rst "Download source file")
-  .pdf

# jax.scipy.linalg.pascal

## Contents

- [`pascal()`](#jax.scipy.linalg.pascal)

# jax.scipy.linalg.pascal[\#](#jax-scipy-linalg-pascal "Link to this heading")

jax.scipy.linalg.pascal(*n*, *kind=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L2969-L3022)[\#](#jax.scipy.linalg.pascal "Link to this definition")  
Create a Pascal matrix approximation of order n.

JAX implementation of [`scipy.linalg.pascal()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.pascal.html#scipy.linalg.pascal "(in SciPy v1.19.0.dev)").

The elements of the Pascal matrix approximate the binomial coefficients. This implementation is not exact as JAX does not support exact factorials.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the size of the matrix to create.

- **kind** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – (optional) must be one of `lower`, `upper`, or `symmetric` (default).

Returns:  
A Pascal matrix of shape `(n,`` ``n)`

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> with jnp.printoptions(precision=3):
    ...   print(jax.scipy.linalg.pascal(3, kind="lower"))
    ...   print(jax.scipy.linalg.pascal(4, kind="upper"))
    ...   print(jax.scipy.linalg.pascal(5))
    [[1. 0. 0.]
     [1. 1. 0.]
     [1. 2. 1.]]
    [[1. 1. 1. 1.]
     [0. 1. 2. 3.]
     [0. 0. 1. 3.]
     [0. 0. 0. 1.]]
    [[ 1.  1.  1.  1.  1.]
     [ 1.  2.  3.  4.  5.]
     [ 1.  3.  6. 10. 15.]
     [ 1.  4. 10. 20. 35.]
     [ 1.  5. 15. 35. 70.]]

[](jax.scipy.linalg.lu_solve.html "previous page")

previous

jax.scipy.linalg.lu_solve

[](jax.scipy.linalg.polar.html "next page")

next

jax.scipy.linalg.polar

Contents

- [`pascal()`](#jax.scipy.linalg.pascal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
