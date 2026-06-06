- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.truncated_normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.truncated_normal.rst "Download source file")
-  .pdf

# jax.nn.initializers.truncated_normal

## Contents

- [`truncated_normal()`](#jax.nn.initializers.truncated_normal)

# jax.nn.initializers.truncated_normal[\#](#jax-nn-initializers-truncated-normal "Link to this heading")

jax.nn.initializers.truncated_normal(*stddev=0.01*, *dtype=None*, *lower=-2.0*, *upper=2.0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L175-L213)[\#](#jax.nn.initializers.truncated_normal "Link to this definition")  
Builds an initializer that returns truncated-normal random arrays.

Parameters:  
- **stddev** (*RealNumeric*) – optional; the standard deviation of the untruncated distribution. Note that this function does not apply the stddev correction as is done in the variancescaling initializers, and users are expected to apply this correction themselves via the stddev arg if they wish to employ it.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – optional; the initializer’s default dtype.

- **lower** (*RealNumeric*) – Float representing the lower bound for truncation. Applied before the output is multiplied by the stddev.

- **upper** (*RealNumeric*) – Float representing the upper bound for truncation. Applied before the output is multiplied by the stddev.

Returns:  
An initializer that returns arrays whose values follow the truncated normal distribution with mean `0` and standard deviation `stddev`, and range \\\rm{lower \* stddev} \< x \< \rm{upper \* stddev}\\.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.truncated_normal(5.0)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[ 2.9047365,  5.2338114,  5.29852  ],
           [-3.836303 , -4.192359 ,  0.6022964]], dtype=float32)

[](jax.nn.initializers.orthogonal.html "previous page")

previous

jax.nn.initializers.orthogonal

[](jax.nn.initializers.uniform.html "next page")

next

jax.nn.initializers.uniform

Contents

- [`truncated_normal()`](#jax.nn.initializers.truncated_normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
