- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.delta_orthogonal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.delta_orthogonal.rst "Download source file")
-  .pdf

# jax.nn.initializers.delta_orthogonal

## Contents

- [`delta_orthogonal()`](#jax.nn.initializers.delta_orthogonal)

# jax.nn.initializers.delta_orthogonal[\#](#jax-nn-initializers-delta-orthogonal "Link to this heading")

jax.nn.initializers.delta_orthogonal(*scale=1.0*, *column_axis=-1*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L645-L707)[\#](#jax.nn.initializers.delta_orthogonal "Link to this definition")  
Builds an initializer for delta orthogonal kernels.

Parameters:  
- **scale** (*RealNumeric*) – the upper bound of the uniform distribution.

- **column_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis that contains the columns that should be orthogonal.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – the default dtype of the weights.

Returns:  
A [delta orthogonal initializer](https://arxiv.org/abs/1806.05393). The shape passed to the initializer must be 3D, 4D, or 5D.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

Examples:

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.delta_orthogonal()
    >>> initializer(jax.random.key(42), (3, 3, 3), jnp.float32)  
    Array([[[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]],

           [[ 0.27858758, -0.7949833 , -0.53887904],
            [ 0.9120717 ,  0.04322892,  0.40774566],
            [-0.30085585, -0.6050892 ,  0.73712474]],

           [[ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ],
            [ 0.        ,  0.        ,  0.        ]]], dtype=float32)

[](jax.nn.initializers.constant.html "previous page")

previous

jax.nn.initializers.constant

[](jax.nn.initializers.glorot_normal.html "next page")

next

jax.nn.initializers.glorot_normal

Contents

- [`delta_orthogonal()`](#jax.nn.initializers.delta_orthogonal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
