- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.multivariate_normal

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.multivariate_normal.rst "Download source file")
-  .pdf

# jax.random.multivariate_normal

## Contents

- [`multivariate_normal()`](#jax.random.multivariate_normal)

# jax.random.multivariate_normal[\#](#jax-random-multivariate-normal "Link to this heading")

jax.random.multivariate_normal(*key*, *mean*, *cov*, *shape=None*, *dtype=None*, *method='cholesky'*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L976-L1039)[\#](#jax.random.multivariate_normal "Link to this definition")  
Sample multivariate normal random values with given mean and covariance.

The values are returned according to the probability density function:

\\f(x;\mu, \Sigma) = (2\pi)^{-k/2} \det(\Sigma)^{-1}e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)}\\

where \\k\\ is the dimension, \\\mu\\ is the mean (given by `mean`) and \\\Sigma\\ is the covariance matrix (given by `cov`).

Parameters:  
- **key** (*ArrayLike*) – a PRNG key used as the random key.

- **mean** (*RealArray*) – a mean vector of shape `(...,`` ``n)`.

- **cov** (*RealArray*) – a positive definite covariance matrix of shape `(...,`` ``n,`` ``n)`. The batch shape `...` must be broadcast-compatible with that of `mean`.

- **shape** (*Shape* *\|* *None*) – optional, a tuple of nonnegative integers specifying the result batch shape; that is, the prefix of the result shape excluding the last axis. Must be broadcast-compatible with `mean.shape[:-1]` and `cov.shape[:-2]`. The default (None) produces a result batch shape by broadcasting together the batch shapes of `mean` and `cov`.

- **dtype** (*DTypeLikeFloat* *\|* *None*) – optional, a float dtype for the returned values (default float64 if jax_enable_x64 is true, otherwise float32).

- **method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – optional, a method to compute the factor of `cov`. Must be one of ‘svd’, ‘eigh’, and ‘cholesky’. Default ‘cholesky’. For singular covariance matrices, use ‘svd’ or ‘eigh’.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*) – Optional. Specifies how the output array should be sharded across devices in multi-device computation. Can be a [`NamedSharding`](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding"), a [`PartitionSpec`](../jax.sharding.html#jax.sharding.PartitionSpec "jax.sharding.PartitionSpec") (`P`), or `None` (default). When specified, the output will be sharded according to the given sharding specification. Primarily used in explicit sharding mode. See the [explicit sharding tutorial](https://docs.jax.dev/en/latest/parallel.html) for more details.

Returns:  
A random array with the specified dtype and shape given by `shape`` ``+`` ``mean.shape[-1:]` if `shape` is not None, or else `broadcast_shapes(mean.shape[:-1],`` ``cov.shape[:-2])`` ``+`` ``mean.shape[-1:]`.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.multinomial.html "previous page")

previous

jax.random.multinomial

[](jax.random.normal.html "next page")

next

jax.random.normal

Contents

- [`multivariate_normal()`](#jax.random.multivariate_normal)

By The JAX authors

© Copyright 2024, The JAX Authors.\
