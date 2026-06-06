- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.hadamard

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.hadamard.rst "Download source file")
-  .pdf

# jax.scipy.linalg.hadamard

## Contents

- [`hadamard()`](#jax.scipy.linalg.hadamard)

# jax.scipy.linalg.hadamard[\#](#jax-scipy-linalg-hadamard "Link to this heading")

jax.scipy.linalg.hadamard(*n*, *dtype=\<class 'int'\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L3136-L3169)[\#](#jax.scipy.linalg.hadamard "Link to this definition")  
Construct an n-by-n Hadamard matrix.

JAX implementation of [`scipy.linalg.hadamard()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.hadamard.html#scipy.linalg.hadamard "(in SciPy v1.19.0.dev)").

For `n` a positive power of 2, the Hadamard matrix \\H_n\\ satisfies \\H_n H_n^T = n I\\. It is defined recursively by the Sylvester construction: \\H_1 = \[\[1\]\]\\, and \\H\_{2m} = \begin{bmatrix} H_m & H_m \\ H_m & -H_m \end{bmatrix}\\.

Parameters:  
- **n** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – size of the matrix. Must be a positive power of 2.

- **dtype** (*DTypeLike*) – output dtype. Defaults to `int`.

Returns:  
A Hadamard matrix of shape `(n,`` ``n)`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> jax.scipy.linalg.hadamard(4)
    Array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]], dtype=int32)

[](jax.scipy.linalg.funm.html "previous page")

previous

jax.scipy.linalg.funm

[](jax.scipy.linalg.helmert.html "next page")

next

jax.scipy.linalg.helmert

Contents

- [`hadamard()`](#jax.scipy.linalg.hadamard)

By The JAX authors

© Copyright 2024, The JAX Authors.\
