- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.zeros

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.zeros.rst "Download source file")
-  .pdf

# jax.nn.initializers.zeros

## Contents

- [`zeros()`](#jax.nn.initializers.zeros)

# jax.nn.initializers.zeros[\#](#jax-nn-initializers-zeros "Link to this heading")

jax.nn.initializers.zeros(*key*, *shape*, *dtype=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L60-L76)[\#](#jax.nn.initializers.zeros "Link to this definition")  
An initializer that returns a constant array full of zeros.

The `key` argument is ignored.

    >>> import jax, jax.numpy as jnp
    >>> jax.nn.initializers.zeros(jax.random.key(42), (2, 3), jnp.float32)
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

Parameters:  
- **key** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **shape** (*core.Shape*)

- **dtype** (*DTypeLikeInexact* *\|* *None*)

- **out_sharding** (*OutShardingType*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.initializers.xavier_uniform.html "previous page")

previous

jax.nn.initializers.xavier_uniform

[](jax.nn.initializers.Initializer.html "next page")

next

jax.nn.initializers.Initializer

Contents

- [`zeros()`](#jax.nn.initializers.zeros)

By The JAX authors

© Copyright 2024, The JAX Authors.\
