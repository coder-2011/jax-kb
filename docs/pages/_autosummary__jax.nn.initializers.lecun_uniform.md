- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.lecun_uniform

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.lecun_uniform.rst "Download source file")
-  .pdf

# jax.nn.initializers.lecun_uniform

## Contents

- [`lecun_uniform()`](#jax.nn.initializers.lecun_uniform)

# jax.nn.initializers.lecun_uniform[\#](#jax-nn-initializers-lecun-uniform "Link to this heading")

jax.nn.initializers.lecun_uniform(*in_axis=-2*, *out_axis=-1*, *batch_axis=()*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L450-L485)[\#](#jax.nn.initializers.lecun_uniform "Link to this definition")  
Builds a Lecun uniform initializer.

A [Lecun uniform initializer](https://arxiv.org/abs/1706.02515) is a specialization of [`jax.nn.initializers.variance_scaling()`](jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling "jax.nn.initializers.variance_scaling") where `scale`` ``=`` ``1.0`, `mode="fan_in"`, and `distribution="uniform"`.

Parameters:  
- **in_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – axis or sequence of axes of the input dimension in the weights array.

- **out_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – axis or sequence of axes of the output dimension in the weights array.

- **batch_axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – axis or sequence of axes in the weight array that should be ignored.

- **dtype** (*DTypeLikeInexact* *\|* *None*) – the dtype of the weights.

Returns:  
An initializer.

Return type:  
[Initializer](jax.nn.initializers.Initializer.html#jax.nn.initializers.Initializer "jax.nn.initializers.Initializer")

Examples:

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.lecun_uniform()
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[ 0.56293887,  0.90433645,  0.9119454 ],
           [-0.71479625, -0.7676109 ,  0.12302713]], dtype=float32)

[](jax.nn.initializers.lecun_normal.html "previous page")

previous

jax.nn.initializers.lecun_normal

[](jax.nn.initializers.normal.html "next page")

next

jax.nn.initializers.normal

Contents

- [`lecun_uniform()`](#jax.nn.initializers.lecun_uniform)

By The JAX authors

© Copyright 2024, The JAX Authors.\
