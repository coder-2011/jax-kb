- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.map

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.map.rst "Download source file")
-  .pdf

# jax.lax.map

## Contents

- [`map()`](#jax.lax.map)

# jax.lax.map[\#](#jax-lax-map "Link to this heading")

jax.lax.map(*f*, *xs*, *\**, *batch_size=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py#L2651-L2724)[\#](#jax.lax.map "Link to this definition")  
Map a function over leading array axes.

Like Python’s builtin map, except inputs and outputs are in the form of stacked arrays. Consider using the [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") transform instead, unless you need to apply a function element by element for reduced memory usage or heterogeneous computation with other control flow primitives.

When `xs` is an array type, the semantics of [`map()`](#jax.lax.map "jax.lax.map") are given by this Python implementation:

    def map(f, xs):
      return np.stack([f(x) for x in xs])

Like [`scan()`](jax.lax.scan.html#jax.lax.scan "jax.lax.scan"), [`map()`](#jax.lax.map "jax.lax.map") is implemented in terms of JAX primitives so many of the same advantages over a Python loop apply: `xs` may be an arbitrary nested pytree type, and the mapped computation is compiled only once.

If `batch_size` is provided, the computation is executed in batches of that size and parallelized using [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap"). This can be used as either a more performant version of `map` or as a memory-efficient version of `vmap`. If the axis is not divisible by the batch size, the remainder is processed in a separate `vmap` and concatenated to the result.

`batch_size=0` is equivalent to applying a `vmap`. That is, it uses a full batch.

    >>> x = jnp.ones((10, 3, 4))
    >>> def f(x):
    ...   print('inner shape:', x.shape)
    ...   return x + 1
    >>> y = lax.map(f, x, batch_size=3)
    inner shape: (3, 4)
    inner shape: (3, 4)
    >>> y.shape
    (10, 3, 4)

In the example above, “inner shape” is printed twice, once while tracing the batched computation and once while tracing the remainder computation.

Parameters:  
- **f** – a Python function to apply element-wise over the first axis or axes of `xs`.

- **xs** – values over which to map along the leading axis.

- **batch_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – (optional) integer specifying the size of the batch for each step to execute in parallel.

Returns:  
Mapped values.

[](jax.lax.fori_loop.html "previous page")

previous

jax.lax.fori_loop

[](jax.lax.scan.html "next page")

next

jax.lax.scan

Contents

- [`map()`](#jax.lax.map)

By The JAX authors

© Copyright 2024, The JAX Authors.\
