- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.constant

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.constant.rst "Download source file")
-  .pdf

# jax.nn.initializers.constant

## Contents

- [`constant()`](#jax.nn.initializers.constant)

# jax.nn.initializers.constant[\#](#jax-nn-initializers-constant "Link to this heading")

jax.nn.initializers.constant(*value*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L95-L118)[\#](#jax.nn.initializers.constant "Link to this definition")  
Builds an initializer that returns arrays full of a constant `value`.

Parameters:  
- **value** (*ArrayLike*) – the constant value with which to fill the initializer.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – optional; the initializer’s default dtype.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.constant(-7)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)
    Array([[-7., -7., -7.],
           [-7., -7., -7.]], dtype=float32)

[](../jax.nn.initializers.html "previous page")

previous

`jax.nn.initializers` module

[](jax.nn.initializers.delta_orthogonal.html "next page")

next

jax.nn.initializers.delta_orthogonal

Contents

- [`constant()`](#jax.nn.initializers.constant)

By The JAX authors

© Copyright 2024, The JAX Authors.\
