- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.normal.rst "Download source file")
-  .pdf

# jax.nn.initializers.normal

## Contents

- [`normal()`](#jax.nn.initializers.normal)

# jax.nn.initializers.normal[\#](#jax-nn-initializers-normal "Link to this heading")

jax.nn.initializers.normal(*stddev=0.01*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L147-L174)[\#](#jax.nn.initializers.normal "Link to this definition")  
Builds an initializer that returns real normally-distributed random arrays.

Parameters:  
- **stddev** (*RealNumeric*) – optional; the standard deviation of the distribution.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – optional; the initializer’s default dtype.

Returns:  
An initializer that returns arrays whose values are normally distributed with mean `0` and standard deviation `stddev`.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.normal(5.0)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[ 3.0613258 ,  5.6129413 ,  5.6866574 ],
           [-4.063663  , -4.4520254 ,  0.63115686]], dtype=float32)

[](jax.nn.initializers.lecun_uniform.html "previous page")

previous

jax.nn.initializers.lecun_uniform

[](jax.nn.initializers.ones.html "next page")

next

jax.nn.initializers.ones

Contents

- [`normal()`](#jax.nn.initializers.normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
