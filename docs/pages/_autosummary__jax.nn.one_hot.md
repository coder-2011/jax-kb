- [](../index.html)
- [API Reference](../jax.html)
- [`jax.nn` module](../jax.nn.html)
- jax.nn.one_hot

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.nn.one_hot.rst "Download source file")
-  .pdf

# jax.nn.one_hot

## Contents

- [`one_hot()`](#jax.nn.one_hot)

# jax.nn.one_hot[\#](#jax-nn-one-hot "Link to this heading")

jax.nn.one_hot(*x*, *num_classes*, *\**, *dtype=None*, *axis=-1*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/nn/functions.py#L741-L778)[\#](#jax.nn.one_hot "Link to this definition")  
One-hot encodes the given indices.

Each index in the input `x` is encoded as a vector of zeros of length `num_classes` with the element at `index` set to one:

    >>> jax.nn.one_hot(jnp.array([0, 1, 2]), 3)
    Array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]], dtype=float32)

Indices outside the range \\\[0, \text{num\\classes})\\ will be encoded as zeros:

    >>> jax.nn.one_hot(jnp.array([-1, 3]), 3)
    Array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

Parameters:  
- **x** (*Any*) – A tensor of indices.

- **num_classes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – Number of classes in the one-hot dimension.

- **dtype** (*Any* *\|* *None*) – optional, a float dtype for the returned values (default `jnp.float_`).

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *AxisName*) – the axis or axes along which the function should be computed.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.nn.standardize.html "previous page")

previous

jax.nn.standardize

[](jax.nn.dot_product_attention.html "next page")

next

jax.nn.dot_product_attention

Contents

- [`one_hot()`](#jax.nn.one_hot)

By The JAX authors

© Copyright 2024, The JAX Authors.\
