- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.he_normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.he_normal.rst "Download source file")
-  .pdf

# jax.nn.initializers.he_normal

## Contents

- [`he_normal()`](#jax.nn.initializers.he_normal)

# jax.nn.initializers.he_normal[\#](#jax-nn-initializers-he-normal "Link to this heading")

jax.nn.initializers.he_normal(*in_axis=-2*, *out_axis=-1*, *batch_axis=()*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L560-L595)[\#](#jax.nn.initializers.he_normal "Link to this definition")  
Builds a He normal initializer (aka Kaiming normal initializer).

A [He normal initializer](https://arxiv.org/abs/1502.01852) is a specialization of [`jax.nn.initializers.variance_scaling()`](jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling "jax.nn.initializers.variance_scaling") where `scale`` ``=`` ``2.0`, `mode="fan_in"`, and `distribution="truncated_normal"`.

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
    >>> initializer = jax.nn.initializers.he_normal()
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[ 0.6604483 ,  1.1900088 ,  1.2047218 ],
           [-0.87225807, -0.95321447,  0.1369438 ]], dtype=float32)

[](jax.nn.initializers.glorot_uniform.html "previous page")

previous

jax.nn.initializers.glorot_uniform

[](jax.nn.initializers.he_uniform.html "next page")

next

jax.nn.initializers.he_uniform

Contents

- [`he_normal()`](#jax.nn.initializers.he_normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
