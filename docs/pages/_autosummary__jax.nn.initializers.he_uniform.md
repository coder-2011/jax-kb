- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.he_uniform

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.he_uniform.rst "Download source file")
-  .pdf

# jax.nn.initializers.he_uniform

## Contents

- [`he_uniform()`](#jax.nn.initializers.he_uniform)

# jax.nn.initializers.he_uniform[\#](#jax-nn-initializers-he-uniform "Link to this heading")

jax.nn.initializers.he_uniform(*in_axis=-2*, *out_axis=-1*, *batch_axis=()*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L522-L557)[\#](#jax.nn.initializers.he_uniform "Link to this definition")  
Builds a He uniform initializer (aka Kaiming uniform initializer).

A [He uniform initializer](https://arxiv.org/abs/1502.01852) is a specialization of [`jax.nn.initializers.variance_scaling()`](jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling "jax.nn.initializers.variance_scaling") where `scale`` ``=`` ``2.0`, `mode="fan_in"`, and `distribution="uniform"`.

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
    >>> initializer = jax.nn.initializers.he_uniform()
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[ 0.79611576,  1.2789248 ,  1.2896855 ],
           [-1.0108745 , -1.0855657 ,  0.17398663]], dtype=float32)

[](jax.nn.initializers.he_normal.html "previous page")

previous

jax.nn.initializers.he_normal

[](jax.nn.initializers.kaiming_normal.html "next page")

next

jax.nn.initializers.kaiming_normal

Contents

- [`he_uniform()`](#jax.nn.initializers.he_uniform)

By The JAX authors

© Copyright 2024, The JAX Authors.\
