- [](../index.html)
- [API Reference](../jax.html)
- jax.jit

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.jit.rst "Download source file")
-  .pdf

# jax.jit

## Contents

- [`jit()`](#jax.jit)

# jax.jit[\#](#jax-jit "Link to this heading")

jax.jit(*fun: [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*, */*, *\**, *in_shardings: Any = UnspecifiedValue*, *out_shardings: Any = UnspecifiedValue*, *static_argnums: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *static_argnames: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| Iterable\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *donate_argnums: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *donate_argnames: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| Iterable\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *keep_unused: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *device: xc.Device \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *backend: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *inline: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *compiler_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), Any\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → pjit.JitWrapped[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L194-L355)[\#](#jax.jit "Link to this definition")\
jax.jit(*\**, *in_shardings: Any = UnspecifiedValue*, *out_shardings: Any = UnspecifiedValue*, *static_argnums: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *static_argnames: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| Iterable\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *donate_argnums: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") \| Sequence\[[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *donate_argnames: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| Iterable\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *keep_unused: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *device: xc.Device \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *backend: [str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*, *inline: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = False*, *compiler_options: [dict](https://docs.python.org/3/library/stdtypes.html#dict "(in Python v3.14)")\[[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"), Any\] \| [None](https://docs.python.org/3/library/constants.html#None "(in Python v3.14)") = None*) → [Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\[\[[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")\], pjit.JitWrapped\]  
Sets up `fun` for just-in-time compilation with XLA.

Parameters:  
- **fun** – Function to be jitted. `fun` should be a pure function. The arguments and return value of `fun` should be arrays, scalar, or (nested) standard Python containers (tuple/list/dict) thereof. Positional arguments indicated by `static_argnums` can be any hashable type. Static arguments are included as part of a compilation cache key, which is why hash and equality operators must be defined. JAX keeps a weak reference to `fun` for use as a compilation cache key, so the object `fun` must be weakly-referenceable. Starting in JAX v0.8.1, when `fun` is omitted, the return value will be a partially-evaluated function to allow the decorator factory pattern (see Examples below).

- **in_shardings** – optional, a `Sharding` or pytree with `Sharding` leaves and structure that is a tree prefix of the positional arguments tuple to `fun`. If provided, the positional arguments passed to `fun` must have shardings that are compatible with `in_shardings` or an error is raised, and the compiled computation has input shardings corresponding to `in_shardings`. If not provided, the compiled computation’s input shardings are inferred from argument shardings.

- **out_shardings** – optional, a `Sharding` or pytree with `Sharding` leaves and structure that is a tree prefix of the output of `fun`. If provided, it has the same effect as applying [`jax.lax.with_sharding_constraint()`](jax.lax.with_sharding_constraint.html#jax.lax.with_sharding_constraint "jax.lax.with_sharding_constraint") to the output of `fun`.

- **static_argnums** –

  optional, an int or collection of ints that specify which positional arguments to treat as static (trace- and compile-time constant).

  Static arguments should be hashable, meaning both `__hash__` and `__eq__` are implemented, and immutable. Otherwise, they can be arbitrary Python objects. Calling the jitted function with different values for these constants will trigger recompilation. Arguments that are not array-like or containers thereof must be marked as static.

  If neither `static_argnums` nor `static_argnames` is provided, no arguments are treated as static. If `static_argnums` is not provided but `static_argnames` is, or vice versa, JAX uses `inspect.signature(fun)` to find any positional arguments that correspond to `static_argnames` (or vice versa). If both `static_argnums` and `static_argnames` are provided, `inspect.signature` is not used, and only actual parameters listed in either `static_argnums` or `static_argnames` will be treated as static.

- **static_argnames** – optional, a string or collection of strings specifying which named arguments to treat as static (compile-time constant). See the comment on `static_argnums` for details. If not provided but `static_argnums` is set, the default is based on calling `inspect.signature(fun)` to find corresponding named arguments.

- **donate_argnums** –

  optional, collection of integers to specify which positional argument buffers can be overwritten by the computation and marked deleted in the caller. It is safe to donate argument buffers if you no longer need them once the computation has started. In some cases XLA can make use of donated buffers to reduce the amount of memory needed to perform a computation, for example recycling one of your input buffers to store a result. You should not reuse buffers that you donate to a computation; JAX will raise an error if you try to. By default, no argument buffers are donated.

  If neither `donate_argnums` nor `donate_argnames` is provided, no arguments are donated. If `donate_argnums` is not provided but `donate_argnames` is, or vice versa, JAX uses `inspect.signature(fun)` to find any positional arguments that correspond to `donate_argnames` (or vice versa). If both `donate_argnums` and `donate_argnames` are provided, `inspect.signature` is not used, and only actual parameters listed in either `donate_argnums` or `donate_argnames` will be donated.

  For more details on buffer donation see the [FAQ](https://docs.jax.dev/en/latest/faq.html#buffer-donation).

- **donate_argnames** – optional, a string or collection of strings specifying which named arguments are donated to the computation. See the comment on `donate_argnums` for details. If not provided but `donate_argnums` is set, the default is based on calling `inspect.signature(fun)` to find corresponding named arguments.

- **keep_unused** – optional boolean. If False (the default), arguments that JAX determines to be unused by fun *may* be dropped from resulting compiled XLA executables. Such arguments will not be transferred to the device nor provided to the underlying executable. If True, unused arguments will not be pruned.

- **device** – This is an experimental feature and the API is likely to change. Optional, the Device the jitted function will run on. (Available devices can be retrieved via [`jax.devices()`](jax.devices.html#jax.devices "jax.devices").) The default is inherited from XLA’s DeviceAssignment logic and is usually to use `jax.devices()[0]`.

- **backend** – This is an experimental feature and the API is likely to change. Optional, a string representing the XLA backend: `'cpu'`, `'gpu'`, or `'tpu'`.

- **inline** – Optional boolean. Specify whether this function should be inlined into enclosing jaxprs. Default False.

Returns:  
A wrapped version of `fun`, set up for just-in-time compilation.

Examples

In the following example, `selu` can be compiled into a single fused kernel by XLA:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def selu(x, alpha=1.67, lmbda=1.05):
    ...   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
    >>>
    >>> key = jax.random.key(0)
    >>> x = jax.random.normal(key, (10,))
    >>> print(selu(x))  
    [-0.54485  0.27744 -0.29255 -0.91421 -0.62452 -0.24748
    -0.85743 -0.78232  0.76827  0.59566 ]

Starting in JAX v0.8.1, [`jit()`](#jax.jit "jax.jit") supports the decorator factory pattern for specifying optional keywords:

    >>> @jax.jit(static_argnames=['n'])
    ... def g(x, n):
    ...   for i in range(n):
    ...     x = x ** 2
    ...   return x
    >>>
    >>> g(jnp.arange(4), 3)
    Array([   0,    1,  256, 6561], dtype=int32)

For compatiblity with older JAX versions, a common pattern is to use [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial "(in Python v3.14)"):

    >>> from functools import partial
    >>>
    >>> @jax.jit(static_argnames=['n'])
    ... def g(x, n):
    ...   for i in range(n):
    ...     x = x ** 2
    ...   return x
    >>>
    >>> g(jnp.arange(4), 3)
    Array([   0,    1,  256, 6561], dtype=int32)

[](jax.transfer_guard.html "previous page")

previous

jax.transfer_guard

[](jax.disable_jit.html "next page")

next

jax.disable_jit

Contents

- [`jit()`](#jax.jit)

By The JAX authors

© Copyright 2024, The JAX Authors.\
