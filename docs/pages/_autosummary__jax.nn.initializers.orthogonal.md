- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.orthogonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.orthogonal.rst "Download source file")
-  .pdf

# jax.nn.initializers.orthogonal

## Contents

- [`orthogonal()`](#jax.nn.initializers.orthogonal)

# jax.nn.initializers.orthogonal[\#](#jax-nn-initializers-orthogonal "Link to this heading")

jax.nn.initializers.orthogonal(*scale=1.0*, *column_axis=-1*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L598-L644)[\#](#jax.nn.initializers.orthogonal "Link to this definition")  
Builds an initializer that returns uniformly distributed orthogonal matrices.

If the shape is not square, the matrices will have orthonormal rows or columns depending on which side is smaller.

Parameters:  
- **scale** (*RealNumeric*) – the upper bound of the uniform distribution.

- **column_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis that contains the columns that should be orthogonal.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – the default dtype of the weights.

Returns:  
An orthogonal initializer.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

Examples:

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.orthogonal()
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[ 3.9026976e-01,  7.2495741e-01, -5.6756169e-01],
           [ 8.8047469e-01, -4.7409311e-01, -1.3157725e-04]],            dtype=float32)

[](jax.nn.initializers.ones.html "previous page")

previous

jax.nn.initializers.ones

[](jax.nn.initializers.truncated_normal.html "next page")

next

jax.nn.initializers.truncated_normal

Contents

- [`orthogonal()`](#jax.nn.initializers.orthogonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
