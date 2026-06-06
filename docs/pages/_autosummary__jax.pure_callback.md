- [](../index.html)
- [API Reference](../jax.html)
- jax.pure_callback

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.pure_callback.rst "Download source file")
-  .pdf

# jax.pure_callback

## Contents

- [`pure_callback()`](#jax.pure_callback)

# jax.pure_callback[\#](#jax-pure-callback "Link to this heading")

jax.pure_callback(*callback*, *result_shape_dtypes*, *\*args*, *sharding=None*, *vmap_method=None*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/callback.py#L261-L394)[\#](#jax.pure_callback "Link to this definition")  
Calls a pure Python callback. Works under [`jit()`](jax.jit.html#jax.jit "jax.jit")/[`vmap()`](jax.vmap.html#jax.vmap "jax.vmap")/etc.

For more explanation, see [External Callbacks](https://docs.jax.dev/en/latest/external-callbacks.html).

`pure_callback` enables calling a Python function in JIT-ed JAX functions. The input `callback` will be passed JAX arrays placed on a local CPU, and it should also return JAX arrays on CPU.

The callback is treated as functionally pure, meaning it has no side-effects and its output value depends only on its argument values. As a consequence, it is safe to be called multiple times (e.g. when transformed by [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") or [`pmap()`](jax.pmap.html#jax.pmap "jax.pmap")), or not to be called at all when e.g. the output of a jit-decorated function has no data dependence on its value. Pure callbacks may also be reordered if data-dependence allows.

Warning

In the context of JAX transformations, Python exceptions should be considered side-effects: this means that intentionally raising an error within a pure_callback breaks the API contract, and the behavior of the resulting program is undefined.

When vmap-ed the behavior will depend on the value of the `vmap_method`.

- Calling [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") on a callback without an explicit `vmap_method` raises a `NotImplementedError`.

- `vmap_method="sequential"` uses [`map()`](jax.lax.map.html#jax.lax.map "jax.lax.map") to loop over the batched arguments, calling `callback` once for each batch element.

- `vmap_method="sequential_unrolled"` is like `sequential`, but the loop is unrolled.

- `vmap_method="expand_dims"` calls `callback` with new axes of size `1` added as the leading dimension unbatched inputs.

- `vmap_method="broadcast_all"` behaves like `expand_dims`, but the inputs are tiled to the expected batched shape.

The current default behavior is to use `vmap_method="sequential"` when not specified, but this behavior is deprecated, and in the future, the default will be to raise a `NotImplementedError` unless `vmap_method` is explicitly specified.

Parameters:  
- **callback** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*) – function to execute on the host. The callback is assumed to be a pure function (i.e. one without side-effects): if an impure function is passed, it may behave in unexpected ways, particularly under transformation. The callable will be passed PyTrees of arrays as arguments, and should return a PyTree of arrays that matches `result_shape_dtypes`.

- **result_shape_dtypes** (*Any*) – pytree whose leaves have `shape` and `dtype` attributes, whose structure matches the expected output of the callback function at runtime. [`jax.ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct") is often used to define leaf values.

- **\*args** (*Any*) – arguments to be passed to the callback function

- **sharding** ([*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – optional sharding that specifies the device from which the callback should be invoked.

- **vmap_method** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – string specifying how the callback transforms under [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") as described above.

- **\*\*kwargs** (*Any*) – keyword arguments to be passed to the callback function

Returns:  
a pytree of [`jax.Array`](jax.Array.html#jax.Array "jax.Array") objects whose structure matches that of  
`result_shape_dtypes`.

Return type:  
result

See also

- [`jax.experimental.io_callback()`](jax.experimental.io_callback.html#jax.experimental.io_callback "jax.experimental.io_callback"): callback designed for impure functions.

- [`jax.debug.callback()`](jax.debug.callback.html#jax.debug.callback "jax.debug.callback"): callback designed for general-purpose debugging.

- [`jax.debug.print()`](jax.debug.print.html#jax.debug.print "jax.debug.print"): callback designed for printing.

Examples

The behavior of `pure_callback` under [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") is controlled by the `vmap_method` argument as described above. It is useful to consider some explicit examples that demonstrate the semantics. For example, consider the following function:

    >>> def callback(x, y):
    ...   print(jnp.shape(x), jnp.shape(y))
    ...   return x + y

    >>> def fun(x, y, *, vmap_method):
    ...   shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    ...   dtype = jnp.result_type(x, y)
    ...   out_type = jax.ShapeDtypeStruct(shape, dtype)
    ...   return jax.pure_callback(callback, out_type, x, y,
    ...                            vmap_method=vmap_method)

Calling this with `vmap_method="expand_dims"` adds a new axis of size `1` to `y`:

    >>> from functools import partial
    >>> x = jnp.arange(4)
    >>> y = 1.0
    >>> jax.vmap(partial(fun, vmap_method="expand_dims"), in_axes=(0, None))(x, y)
    (4,) (1,)
    Array([1., 2., 3., 4.], dtype=float32)

Whereas, `vmap_method="broadcast_all"` adds an axis of size `4` to `y`:

    >>> jax.vmap(partial(fun, vmap_method="broadcast_all"),
    ...          in_axes=(0, None))(x, y)
    (4,) (4,)
    Array([1., 2., 3., 4.], dtype=float32)

[](jax.Array.mT.html "previous page")

previous

jax.Array.mT

[](jax.experimental.io_callback.html "next page")

next

jax.experimental.io_callback

Contents

- [`pure_callback()`](#jax.pure_callback)

By The JAX authors

© Copyright 2024, The JAX Authors.\
