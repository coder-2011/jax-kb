- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.stats.gaussian_kde.resample

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.stats.gaussian_kde.resample.rst "Download source file")
-  .pdf

# jax.scipy.stats.gaussian_kde.resample

## Contents

- [`gaussian_kde.resample()`](#jax.scipy.stats.gaussian_kde.resample)

# jax.scipy.stats.gaussian_kde.resample[\#](#jax-scipy-stats-gaussian-kde-resample "Link to this heading")

gaussian_kde.resample(*key*, *shape=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/stats/kde.py#L186-L207)[\#](#jax.scipy.stats.gaussian_kde.resample "Link to this definition")  
Randomly sample a dataset from the estimated pdf

Parameters:  
- **key** – a PRNG key used as the random key.

- **shape** – optional, a tuple of nonnegative integers specifying the result batch shape; that is, the prefix of the result shape excluding the last axis.

Returns:  
The resampled dataset as an array with shape (d,) + shape.

[](jax.scipy.stats.gaussian_kde.integrate_kde.html "previous page")

previous

jax.scipy.stats.gaussian_kde.integrate_kde

[](jax.scipy.stats.gaussian_kde.pdf.html "next page")

next

jax.scipy.stats.gaussian_kde.pdf

Contents

- [`gaussian_kde.resample()`](#jax.scipy.stats.gaussian_kde.resample)

By The JAX authors

© Copyright 2024, The JAX Authors.\
