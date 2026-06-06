- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.uniform

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.uniform.rst "Download source file")
-  .pdf

# jax.nn.initializers.uniform

## Contents

- [`uniform()`](#jax.nn.initializers.uniform)

# jax.nn.initializers.uniform[\#](#jax-nn-initializers-uniform "Link to this heading")

jax.nn.initializers.uniform(*scale=0.01*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L119-L146)[\#](#jax.nn.initializers.uniform "Link to this definition")  
Builds an initializer that returns real uniformly-distributed random arrays.

Parameters:  
- **scale** (*RealNumeric*) – optional; the upper bound of the random distribution.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – optional; the initializer’s default dtype.

Returns:  
An initializer that returns arrays whose values are uniformly distributed in the range `[0,`` ``scale)`.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.uniform(10.0)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[7.298188 , 8.691938 , 8.7230015],
           [2.0818567, 1.8662417, 5.5022564]], dtype=float32)

[](jax.nn.initializers.truncated_normal.html "previous page")

previous

jax.nn.initializers.truncated_normal

[](jax.nn.initializers.variance_scaling.html "next page")

next

jax.nn.initializers.variance_scaling

Contents

- [`uniform()`](#jax.nn.initializers.uniform)

By The JAX authors

© Copyright 2024, The JAX Authors.\
