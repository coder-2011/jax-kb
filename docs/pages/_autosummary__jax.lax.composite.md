- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.composite

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.composite.rst "Download source file")
-  .pdf

# jax.lax.composite

## Contents

- [`composite()`](#jax.lax.composite)

# jax.lax.composite[\#](#jax-lax-composite "Link to this heading")

jax.lax.composite(*decomposition*, *name*, *version=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L1835-L1936)[\#](#jax.lax.composite "Link to this definition")  
Composite with semantics defined by the decomposition function.

A composite is a higher-order JAX function that encapsulates an operation made up (composed) of other JAX functions. The semantics of the op are implemented by the `decomposition` function. In other words, the defined composite function can be replaced with its decomposed implementation without changing the semantics of the encapsulated operation.

The compiler can recognize specific composite operations by their `name`, `version`, `kwargs`, and dtypes to emit more efficient code, potentially leveraging hardware-specific instructions or optimizations. If the compiler doesn’t recognize the composite, it falls back to compiling the `decomposition` function.

Consider a “tangent” composite operation. Its `decomposition` function could be implemented as `sin(x)`` ``/`` ``cos(x)`. A hardware-aware compiler could recognize the “tangent” composite and emit a single `tangent` instruction instead of three separate instructions (`sin`, `divide`, and `cos`). For hardware without dedicated tangent support, it would fall back to compiling the decomposition.

This is useful for preserving high-level abstractions that would otherwise be lost while lowering, which allows for easier pattern-matching in low-level IR.

Parameters:  
- **decomposition** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – function that implements the semantics of the composite op.

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – name of the encapsulated operation.

- **version** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional int to indicate semantic changes to the composite.

Returns:  
Returns a composite function. Note that positional arguments to this function should be interpreted as inputs and keyword arguments should be interpreted as attributes of the op. Any keyword arguments that are passed with `None` as a value will be omitted from the `composite_attributes`.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")

Examples

Tangent kernel:

    >>> def my_tangent_composite(x):
    ...   return lax.composite(
    ...     lambda x: lax.sin(x) / lax.cos(x), name="my.tangent"
    ...   )(x)
    >>>
    >>> pi = jnp.pi
    >>> x = jnp.array([0.0, pi / 4, 3 * pi / 4, pi])
    >>> with jnp.printoptions(precision=3, suppress=True):
    ...   print(my_tangent_composite(x))
    ...   print(lax.tan(x))
    [ 0.  1. -1.  0.]
    [ 0.  1. -1.  0.]

The recommended way to create composites is via a decorator. Use `/` and `*` in the function signature to be explicit about positional and keyword arguments, respectively:

    >>> @partial(lax.composite, name="my.softmax")
    ... def my_softmax_composite(x, /, *, axis):
    ...   return jax.nn.softmax(x, axis)

[](jax.lax.complex.html "previous page")

previous

jax.lax.complex

[](jax.lax.concatenate.html "next page")

next

jax.lax.concatenate

Contents

- [`composite()`](#jax.lax.composite)

By The JAX authors

© Copyright 2024, The JAX Authors.\
