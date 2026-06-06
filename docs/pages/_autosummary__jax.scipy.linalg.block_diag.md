- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.linalg.block_diag

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.linalg.block_diag.rst "Download source file")
-  .pdf

# jax.scipy.linalg.block_diag

## Contents

- [`block_diag()`](#jax.scipy.linalg.block_diag)

# jax.scipy.linalg.block_diag[\#](#jax-scipy-linalg-block-diag "Link to this heading")

jax.scipy.linalg.block_diag(*\*arrs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/linalg.py#L1628-L1673)[\#](#jax.scipy.linalg.block_diag "Link to this definition")  
Create a block diagonal matrix from input arrays.

JAX implementation of [`scipy.linalg.block_diag()`](https://scipy.github.io/devdocs/reference/generated/scipy.linalg.block_diag.html#scipy.linalg.block_diag "(in SciPy v1.19.0.dev)").

Parameters:  
**\*arrs** (*ArrayLike*) – arrays of at most two dimensions

Returns:  
2D block-diagonal array constructed by placing the input arrays along the diagonal.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

Examples

    >>> A = jnp.ones((1, 1))
    >>> B = jnp.ones((2, 2))
    >>> C = jnp.ones((3, 3))
    >>> jax.scipy.linalg.block_diag(A, B, C)
    Array([[1., 0., 0., 0., 0., 0.],
           [0., 1., 1., 0., 0., 0.],
           [0., 1., 1., 0., 0., 0.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.],
           [0., 0., 0., 1., 1., 1.]], dtype=float32)

[](jax.scipy.interpolate.RegularGridInterpolator.html "previous page")

previous

jax.scipy.interpolate.RegularGridInterpolator

[](jax.scipy.linalg.cho_factor.html "next page")

next

jax.scipy.linalg.cho_factor

Contents

- [`block_diag()`](#jax.scipy.linalg.block_diag)

By The JAX authors

© Copyright 2024, The JAX Authors.\
