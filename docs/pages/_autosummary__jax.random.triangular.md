- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.triangular

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.triangular.rst "Download source file")
-  .pdf

# jax.random.triangular

## Contents

- [`triangular()`](#jax.random.triangular)

# jax.random.triangular[\#](#jax-random-triangular "Link to this heading")

jax.random.triangular(*key*, *left*, *mode*, *right*, *shape=None*, *dtype=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L3000-L3056)[\#](#jax.random.triangular "Link to this definition")  
Sample Triangular random values with given shape and float dtype.

The values are returned according to the probability density function:

\\\begin{split}f(x; a, b, c) = \frac{2}{c-a} \left\\ \begin{array}{ll} \frac{x-a}{b-a} & a \leq x \leq b \\ \frac{c-x}{c-b} & b \leq x \leq c \end{array} \right.\end{split}\\

on the domain \\a \leq x \leq c\\.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **left** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the lower limit parameter of the distribution.

- **mode** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the peak value parameter of the distribution, value must fulfill the condition `left`` ``<=`` ``mode`` ``<=`` ``right`.

- **right** (*RealArray*) – a float or array of floats broadcast-compatible with `shape` representing the upper limit parameter of the distribution, must be larger than `left`.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result shape. Must be broadcast-compatible with `left`,\`\`mode\`\` and `right`. The default (None) produces a result shape equal to `left.shape`, `mode.shape` and `right.shape`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – optional, specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and with shape given by `shape` if `shape` is not None, or else by `left.shape`, `mode.shape` and `right.shape`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.t.html "previous page")

previous

jax.random.t

[](jax.random.truncated_normal.html "next page")

next

jax.random.truncated_normal

Contents

- [`triangular()`](#jax.random.triangular)

By The JAX authors

© Copyright 2024, The JAX Authors.\
