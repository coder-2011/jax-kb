- [](../index.html)
- [API Reference](../jax.html)
- [`jax.scipy` module](../jax.scipy.html)
- jax.scipy.cluster.vq.vq

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.scipy.cluster.vq.vq.rst "Download source file")
-  .pdf

# jax.scipy.cluster.vq.vq

## Contents

- [`vq()`](#jax.scipy.cluster.vq.vq)

# jax.scipy.cluster.vq.vq[\#](#jax-scipy-cluster-vq-vq "Link to this heading")

jax.scipy.cluster.vq.vq(*obs*, *code_book*, *check_finite=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/scipy/cluster/vq.py#L25-L76)[\#](#jax.scipy.cluster.vq.vq "Link to this definition")  
Assign codes from a code book to a set of observations.

JAX implementation of [`scipy.cluster.vq.vq()`](https://scipy.github.io/devdocs/reference/generated/scipy.cluster.vq.vq.html#scipy.cluster.vq.vq "(in SciPy v1.19.0.dev)").

Assigns each observation vector in `obs` to a code from `code_book` based on the nearest Euclidean distance.

Parameters:  
- **obs** (*ArrayLike*) – array of observation vectors of shape `(M,`` ``N)`. Each row represents a single observation. If `obs` is one-dimensional, then each entry is treated as a length-1 observation.

- **code_book** (*ArrayLike*) – array of codes with shape `(K,`` ``N)`. Each row represents a single code vector. If `code_book` is one-dimensional, then each entry is treated as a length-1 code.

- **check_finite** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused in JAX

Returns:  
A tuple of arrays `(code,`` ``dist)`

- `code` is an integer array of shape `(M,)` containing indices `0`` ``<=`` ``i`` ``<`` ``K` of the closest entry in `code_book` for the given entry in `obs`.

- `dist` is a float array of shape `(M,)` containing the euclidean distance between each observation and the nearest code.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

Examples

    >>> obs = jnp.array([[1.1, 2.1, 3.1],
    ...                  [5.9, 4.8, 6.2]])
    >>> code_book = jnp.array([[1., 2., 3.],
    ...                        [2., 3., 4.],
    ...                        [3., 4., 5.],
    ...                        [4., 5., 6.]])
    >>> codes, distances = jax.scipy.cluster.vq.vq(obs, code_book)
    >>> print(codes)
    [0 3]
    >>> print(distances)
    [0.17320499 1.9209373 ]

[](../jax.scipy.html "previous page")

previous

`jax.scipy` module

[](jax.scipy.fft.dct.html "next page")

next

jax.scipy.fft.dct

Contents

- [`vq()`](#jax.scipy.cluster.vq.vq)

By The JAX authors

© Copyright 2024, The JAX Authors.\
