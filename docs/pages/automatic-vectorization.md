- [](index.html)
- [JAX 101](jax-101.html)
- Automatic vectorization

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/automatic-vectorization.md "Download source file")
-  .pdf

# Automatic vectorization

## Contents

- [Manual vectorization](#manual-vectorization)
- [Automatic vectorization](#id2)
- [Combining transformations](#combining-transformations)

# Automatic vectorization[\#](#automatic-vectorization "Link to this heading")

In the previous section we discussed JIT compilation via the [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") function. This notebook discusses another of JAX’s transforms: vectorization via [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap").

## Manual vectorization[\#](#manual-vectorization "Link to this heading")

Consider the following simple code that computes the convolution of two one-dimensional vectors:

    import jax
    import jax.numpy as jnp

    x = jnp.arange(5)
    w = jnp.array([2., 3., 4.])

    def convolve(x, w):
      output = []
      for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
      return jnp.array(output)

    convolve(x, w)

    Array([11., 20., 29.], dtype=float32)

Suppose we would like to apply this function to a batch of weights `w` to a batch of vectors `x`.

    xs = jnp.stack([x, x])
    ws = jnp.stack([w, w])

The most naive option would be to simply loop over the batch in Python:

    def manually_batched_convolve(xs, ws):
      output = []
      for i in range(xs.shape[0]):
        output.append(convolve(xs[i], ws[i]))
      return jnp.stack(output)

    manually_batched_convolve(xs, ws)

    Array([[11., 20., 29.],
           [11., 20., 29.]], dtype=float32)

This produces the correct result, however it is not very efficient.

In order to batch the computation efficiently, you would normally have to rewrite the function manually to ensure it is done in vectorized form. This is not particularly difficult to implement, but does involve changing how the function treats indices, axes, and other parts of the input.

For example, we could manually rewrite `convolve()` to support vectorized computation across the batch dimension as follows:

    def manually_vectorized_convolve(xs, ws):
      output = []
      for i in range(1, xs.shape[-1] -1):
        output.append(jnp.sum(xs[:, i-1:i+2] * ws, axis=1))
      return jnp.stack(output, axis=1)

    manually_vectorized_convolve(xs, ws)

    Array([[11., 20., 29.],
           [11., 20., 29.]], dtype=float32)

Such re-implementation can be messy and error-prone as the complexity of a function increases; fortunately JAX provides another way.

## Automatic vectorization[\#](#id2 "Link to this heading")

In JAX, the [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") transformation is designed to generate such a vectorized implementation of a function automatically:

    auto_batch_convolve = jax.vmap(convolve)

    auto_batch_convolve(xs, ws)

    Array([[11., 20., 29.],
           [11., 20., 29.]], dtype=float32)

It does this by tracing the function similarly to [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), and automatically adding batch axes at the beginning of each input.

If the batch dimension is not the first, you may use the `in_axes` and `out_axes` arguments to specify the location of the batch dimension in inputs and outputs. These may be an integer if the batch axis is the same for all inputs and outputs, or lists, otherwise.

    auto_batch_convolve_v2 = jax.vmap(convolve, in_axes=1, out_axes=1)

    xst = jnp.transpose(xs)
    wst = jnp.transpose(ws)

    auto_batch_convolve_v2(xst, wst)

    Array([[11., 11.],
           [20., 20.],
           [29., 29.]], dtype=float32)

[`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") also supports the case where only one of the arguments is batched: for example, if you would like to convolve to a single set of weights `w` with a batch of vectors `x`; in this case the `in_axes` argument can be set to `None`:

    batch_convolve_v3 = jax.vmap(convolve, in_axes=[0, None])

    batch_convolve_v3(xs, w)

    Array([[11., 20., 29.],
           [11., 20., 29.]], dtype=float32)

## Combining transformations[\#](#combining-transformations "Link to this heading")

As with all JAX transformations, [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") and [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") are designed to be composable, which means you can wrap a vmapped function with `jit`, or a jitted function with `vmap`, and everything will work correctly:

    jitted_batch_convolve = jax.jit(auto_batch_convolve)

    jitted_batch_convolve(xs, ws)

    Array([[11., 20., 29.],
           [11., 20., 29.]], dtype=float32)

It is sometimes useful to compose `vmap` with itself. For instance, we can succinctly express the pairwise evaluation of a function using two nested applications of `vmap`:

    def pairwise(f, xs):
      return jax.vmap(lambda x: jax.vmap(lambda y: f(x, y))(xs))(xs)

Suppose `xs` is a matrix whose M rows are each N-dimensional points, and that `dist(x,`` ``y)` computes the distance between two N-dimensional vectors. Then `pairwise(dist,`` ``xs)` is the M-by-M matrix of pairwise distances among `xs`.

[](jit-compilation.html "previous page")

previous

Just-in-time compilation

[](automatic-differentiation.html "next page")

next

Automatic differentiation

Contents

- [Manual vectorization](#manual-vectorization)
- [Automatic vectorization](#id2)
- [Combining transformations](#combining-transformations)

By The JAX authors

© Copyright 2024, The JAX Authors.\
