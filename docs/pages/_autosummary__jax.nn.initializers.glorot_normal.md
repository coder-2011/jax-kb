- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- [`jax.nn.initializers` module](../jax.nn.initializers.html)
- jax.nn.initializers.glorot_normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.initializers.glorot_normal.rst "Download source file")
-  .pdf

# jax.nn.initializers.glorot_normal

## Contents

- [`glorot_normal()`](#jax.nn.initializers.glorot_normal)

# jax.nn.initializers.glorot_normal[\#](#jax-nn-initializers-glorot-normal "Link to this heading")

jax.nn.initializers.glorot_normal(*in_axis=-2*, *out_axis=-1*, *batch_axis=()*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L412-L447)[\#](#jax.nn.initializers.glorot_normal "Link to this definition")  
Builds a Glorot normal initializer (aka Xavier normal initializer).

A [Glorot normal initializer](http://proceedings.mlr.press/v9/glorot10a.html) is a specialization of [`jax.nn.initializers.variance_scaling()`](jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling "jax.nn.initializers.variance_scaling") where `scale`` ``=`` ``1.0`, `mode="fan_avg"`, and `distribution="truncated_normal"`.

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
    >>> initializer = jax.nn.initializers.glorot_normal()
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  
    Array([[ 0.41770416,  0.75262755,  0.7619329 ],
           [-0.5516644 , -0.6028657 ,  0.08661086]], dtype=float32)

[](jax.nn.initializers.delta_orthogonal.html "previous page")

previous

jax.nn.initializers.delta_orthogonal

[](jax.nn.initializers.glorot_uniform.html "next page")

next

jax.nn.initializers.glorot_uniform

Contents

- [`glorot_normal()`](#jax.nn.initializers.glorot_normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
