- [](../index.html)
- [API Reference](../jax.html)
- jax.experimental.io_callback

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.io_callback.rst "Download source file")
-  .pdf

# jax.experimental.io_callback

## Contents

- [`io_callback()`](#jax.experimental.io_callback)

# jax.experimental.io_callback[\#](#jax-experimental-io-callback "Link to this heading")

jax.experimental.io_callback(*callback*, *result_shape_dtypes*, *\*args*, *sharding=None*, *ordered=False*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/callback.py#L538-L587)[\#](#jax.experimental.io_callback "Link to this definition")  
Calls an impure Python callback.

For more explanation, see [External Callbacks](https://docs.jax.dev/en/latest/notebooks/external_callbacks.html).

Parameters:  
- **callback** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*) – function to execute on the host. It is assumed to be an impure function. If `callback` is pure, using [`jax.pure_callback()`](jax.pure_callback.html#jax.pure_callback "jax.pure_callback") instead may lead to more efficient execution.

- **result_shape_dtypes** (*Any*) – pytree whose leaves have `shape` and `dtype` attributes, whose structure matches the expected output of the callback function at runtime. [`jax.ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct") is often used to define leaf values.

- **\*args** (*Any*) – arguments to be passed to the callback function

- **sharding** ([*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*) – optional sharding that specifies the device from which the callback should be invoked.

- **ordered** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – boolean specifying whether sequential calls to callback must be ordered.

- **\*\*kwargs** (*Any*) – keyword arguments to be passed to the callback function

Returns:  
a pytree of [`jax.Array`](jax.Array.html#jax.Array "jax.Array") objects whose structure matches that of  
`result_shape_dtypes`.

Return type:  
result

See also

- [`jax.pure_callback()`](jax.pure_callback.html#jax.pure_callback "jax.pure_callback"): callback designed for pure functions.

- [`jax.debug.callback()`](jax.debug.callback.html#jax.debug.callback "jax.debug.callback"): callback designed for general-purpose debugging.

- [`jax.debug.print()`](jax.debug.print.html#jax.debug.print "jax.debug.print"): callback designed for printing.

[](jax.pure_callback.html "previous page")

previous

jax.pure_callback

[](jax.Device.html "next page")

next

jax.Device

Contents

- [`io_callback()`](#jax.experimental.io_callback)

By The JAX authors

© Copyright 2024, The JAX Authors.\
