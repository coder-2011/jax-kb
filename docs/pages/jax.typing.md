- [](index.html)
- [API Reference](jax.html)
- `jax.typing` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.typing.rst "Download source file")
-  .pdf

# jax.typing module

## Contents

- [JAX Typing Best Practices](#jax-typing-best-practices)
- [List of Members](#list-of-members)

# `jax.typing` module[\#](#module-jax.typing "Link to this heading")

The JAX typing module is where JAX-specific static type annotations live. This submodule is a work in progress; to see the proposal behind the types exported here, see [https://docs.jax.dev/en/latest/jep/12049-type-annotations.html](https://docs.jax.dev/en/latest/jep/12049-type-annotations.html).

The currently-available types are:

- [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array"): annotation for any JAX array or tracer (i.e. representations of arrays within JAX transforms).

- [`jax.typing.ArrayLike`](_autosummary/jax.typing.ArrayLike.html#jax.typing.ArrayLike "jax.typing.ArrayLike"): annotation for any value that is safe to implicitly cast to a JAX array; this includes [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array"), [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)"), as well as Python builtin numeric values (e.g. [`int`](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"), [`float`](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)"), etc.) and numpy scalar values (e.g. `numpy.int32`, `numpy.float64`, etc.)

- [`jax.typing.DTypeLike`](_autosummary/jax.typing.DTypeLike.html#jax.typing.DTypeLike "jax.typing.DTypeLike"): annotation for any value that can be cast to a JAX-compatible dtype; this includes strings (e.g. ‘float32’, ‘int32’), scalar types (e.g. float, np.float32), dtypes (e.g. np.dtype(‘float32’)), or objects with a dtype attribute (e.g. jnp.float32, jnp.int32).

We may add additional types here in future releases.

## JAX Typing Best Practices[\#](#jax-typing-best-practices "Link to this heading")

When annotating JAX arrays in public API functions, we recommend using [`ArrayLike`](_autosummary/jax.typing.ArrayLike.html#jax.typing.ArrayLike "jax.typing.ArrayLike") for array inputs, and [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") for array outputs.

For example, your function might look like this:

    import numpy as np
    import jax.numpy as jnp
    from jax import Array
    from jax.typing import ArrayLike

    def my_function(x: ArrayLike) -> Array:
      # Runtime type validation, Python 3.10 or newer:
      if not isinstance(x, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {x}")
      # Runtime type validation, any Python version:
      if not (isinstance(x, (np.ndarray, Array)) or np.isscalar(x)):
        raise TypeError(f"Expected arraylike input; got {x}")

      # Convert input to jax.Array:
      x_arr = jnp.asarray(x)

      # ... do some computation; JAX functions will return Array types:
      result = x_arr.sum(0) / x_arr.shape[0]

      # return an Array
      return result

Most of JAX’s public APIs follow this pattern. Note in particular that we recommend JAX functions to not accept sequences such as [`list`](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)") or [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)") in place of arrays, as this can cause extra overhead in JAX transforms like [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") and can behave in unexpected ways with batch-wise transforms like [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") or [`jax.pmap()`](_autosummary/jax.pmap.html#jax.pmap "jax.pmap"). For more information on this, see [Non-array inputs NumPy vs JAX](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#non-array-inputs-numpy-vs-jax)

## List of Members[\#](#list-of-members "Link to this heading")

|  |  |
|----|----|
| [`ArrayLike`](_autosummary/jax.typing.ArrayLike.html#jax.typing.ArrayLike "jax.typing.ArrayLike") | Type annotation for JAX array-like objects. |
| [`DTypeLike`](_autosummary/jax.typing.DTypeLike.html#jax.typing.DTypeLike "jax.typing.DTypeLike") | alias of [`str`](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") \| [`type`](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")\[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any "(in Python v3.14)")\] \| [`dtype`](_autosummary/jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype") \| `SupportsDType` |

[](_autosummary/jax.tree_util.tree_unflatten.html "previous page")

previous

jax.tree_util.tree_unflatten

[](_autosummary/jax.typing.ArrayLike.html "next page")

next

jax.typing.ArrayLike

Contents

- [JAX Typing Best Practices](#jax-typing-best-practices)
- [List of Members](#list-of-members)

By The JAX authors

© Copyright 2024, The JAX Authors.\
