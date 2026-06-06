- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.ones

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.ones.rst "Download source file")
-  .pdf

# jax.nn.initializers.ones

## Contents

- [`ones()`](#jax.nn.initializers.ones)

# jax.nn.initializers.ones[\#](#jax-nn-initializers-ones "Link to this heading")

jax.nn.initializers.ones(*key*, *shape*, *dtype=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L77-L94)[\#](#jax.nn.initializers.ones "Link to this definition")  
An initializer that returns a constant array full of ones.

The `key` argument is ignored.

    >>> import jax, jax.numpy as jnp
    >>> jax.nn.initializers.ones(jax.random.key(42), (3, 2), jnp.float32)
    Array([[1., 1.],
           [1., 1.],
           [1., 1.]], dtype=float32)

Parameters:  
- **key** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **shape** (*core.Shape*)

- **dtype** (*DTypeLikeInexact* *\|* *None*)

- **out_sharding** (*OutShardingType*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.initializers.normal.html "previous page")

previous

jax.nn.initializers.normal

[](jax.nn.initializers.orthogonal.html "next page")

next

jax.nn.initializers.orthogonal

Contents

- [`ones()`](#jax.nn.initializers.ones)

By The JAX authors

© Copyright 2024, The JAX Authors.\
