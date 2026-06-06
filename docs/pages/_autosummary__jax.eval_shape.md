- [](../index.html)
- [API Reference](../jax.html)
- jax.eval_shape

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.eval_shape.rst "Download source file")
-  .pdf

# jax.eval_shape

## Contents

- [`eval_shape()`](#jax.eval_shape)

# jax.eval_shape[\#](#jax-eval-shape "Link to this heading")

jax.eval_shape(*fun*, *\*args*, *\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L2310-L2380)[\#](#jax.eval_shape "Link to this definition")  
Compute the shape/dtype of `fun` without any FLOPs.

This utility function is useful for performing shape inference. Its input/output behavior is defined by:

    def eval_shape(fun, *args, **kwargs):
      out = fun(*args, **kwargs)
      return jax.tree_util.tree_map(jax.ShapeDtypeStruct.like, out)

But instead of applying `fun` directly, which might be expensive, it uses JAX’s abstract interpretation machinery to evaluate the shapes without doing any FLOPs.

Using [`eval_shape()`](#jax.eval_shape "jax.eval_shape") can also catch shape errors, and will raise same shape errors as evaluating `fun(*args,`` ``**kwargs)`.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – The function whose output shape should be evaluated.

- **\*args** – a positional argument tuple of arrays, scalars, or (nested) standard Python containers (tuples, lists, dicts, namedtuples, i.e. pytrees) of those types. Since only the `shape` and `dtype` attributes are accessed, one can use [`jax.ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct") or another container that duck-types as ndarrays (note however that duck-typed objects cannot be namedtuples because those are treated as standard Python containers).

- **\*\*kwargs** – a keyword argument dict of arrays, scalars, or (nested) standard Python containers (pytrees) of those types. As in `args`, array values need only be duck-typed to have `shape` and `dtype` attributes.

Returns:  
a nested PyTree containing [`jax.ShapeDtypeStruct`](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct") objects as leaves.

Return type:  
out

For example:

    >>> import jax
    >>> import jax.numpy as jnp
    >>>
    >>> f = lambda A, x: jnp.tanh(jnp.dot(A, x))
    >>> A = jax.ShapeDtypeStruct((2000, 3000), jnp.float32)
    >>> x = jax.ShapeDtypeStruct((3000, 1000), jnp.float32)
    >>> out = jax.eval_shape(f, A, x)  # no FLOPs performed
    >>> print(out.shape)
    (2000, 1000)
    >>> print(out.dtype)
    float32

All arguments passed via [`eval_shape()`](#jax.eval_shape "jax.eval_shape") will be treated as dynamic; static arguments can be included via closure, for example using [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial "(in Python v3.14)"):

    >>> import jax
    >>> from jax import lax
    >>> from functools import partial
    >>> import jax.numpy as jnp
    >>>
    >>> x = jax.ShapeDtypeStruct((1, 1, 28, 28), jnp.float32)
    >>> kernel = jax.ShapeDtypeStruct((32, 1, 3, 3), jnp.float32)
    >>>
    >>> conv_same = partial(lax.conv_general_dilated, window_strides=(1, 1), padding="SAME")
    >>> out = jax.eval_shape(conv_same, x, kernel)
    >>> print(out.shape)
    (1, 32, 28, 28)
    >>> print(out.dtype)
    float32

[](jax.make_jaxpr.html "previous page")

previous

jax.make_jaxpr

[](jax.ShapeDtypeStruct.html "next page")

next

jax.ShapeDtypeStruct

Contents

- [`eval_shape()`](#jax.eval_shape)

By The JAX authors

© Copyright 2024, The JAX Authors.\
