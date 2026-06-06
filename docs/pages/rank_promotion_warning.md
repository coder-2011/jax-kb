- [](index.html)
- [Notes](notes.html)
- Rank promotion warning

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/rank_promotion_warning.rst "Download source file")
-  .pdf

# Rank promotion warning

# Rank promotion warning[\#](#rank-promotion-warning "Link to this heading")

[NumPy broadcasting rules](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules) allow the automatic promotion of arguments from one rank (number of array axes) to another. This behavior can be convenient when intended but can also lead to surprising bugs where a silent rank promotion masks an underlying shape error.

Here’s an example of rank promotion:

    >>> from jax import numpy as jnp
    >>> x = jnp.arange(12).reshape(4, 3)
    >>> y = jnp.array([0, 1, 0])
    >>> x + y
    Array([[ 0,  2,  2],
           [ 3,  5,  5],
           [ 6,  8,  8],
           [ 9, 11, 11]], dtype=int32)

To avoid potential surprises, `jax.numpy` is configurable so that expressions requiring rank promotion can lead to a warning, error, or can be allowed just like regular NumPy. The configuration option is named `jax_numpy_rank_promotion` and it can take on string values `allow`, `warn`, and `raise`. The default setting is `allow`, which allows rank promotion without warning or error. The `raise` setting raises an error on rank promotion, and `warn` raises a warning on the first occurrence of rank promotion.

Rank promotion can be enabled or disabled locally with the [`jax.numpy_rank_promotion()`](_autosummary/jax.numpy_rank_promotion.html#jax.numpy_rank_promotion "jax.numpy_rank_promotion") context manager:

    with jax.numpy_rank_promotion("warn"):
      z = x + y

This configuration can also be set globally in several ways. One is by using `jax.config` in your code:

    import jax
    jax.config.update("jax_numpy_rank_promotion", "warn")

You can also set the option using the environment variable `JAX_NUMPY_RANK_PROMOTION`, for example as `JAX_NUMPY_RANK_PROMOTION='warn'`. Finally, when using `absl-py` the option can be set with a command-line flag.

[](gpu_memory_allocation.html "previous page")

previous

GPU memory allocation

[](type_promotion.html "next page")

next

Type promotion semantics

By The JAX authors

© Copyright 2024, The JAX Authors.\
