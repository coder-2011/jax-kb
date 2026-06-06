- [](index.html)
- Key concepts

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/key-concepts.md "Download source file")
-  .pdf

# Key concepts

## Contents

- [Transformations](#transformations)
- [Tracing](#tracing)
- [Jaxprs](#jaxprs)
- [Pytrees](#pytrees)
- [JAX API layering: NumPy, lax & XLA](#jax-api-layering-numpy-lax-xla)

# Key concepts[\#](#key-concepts "Link to this heading")

This section briefly introduces some key concepts of the JAX package.

## Transformations[\#](#transformations "Link to this heading")

Along with functions to operate on arrays, JAX includes a number of [transformations](glossary.html#term-transformation) which operate on JAX functions. These include

- [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"): Just-in-time (JIT) compilation; see [Just-in-time compilation](jit-compilation.html#jit-compilation)

- [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"): Vectorizing transform; see [Automatic vectorization](automatic-vectorization.html#automatic-vectorization)

- [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"): Gradient transform; see [Automatic differentiation](automatic-differentiation.html#automatic-differentiation)

as well as several others. Transformations accept a function as an argument, and return a new transformed function. For example, here’s how you might JIT-compile a simple SELU function:

    import jax
    import jax.numpy as jnp

    def selu(x, alpha=1.67, lambda_=1.05):
      return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

    selu_jit = jax.jit(selu)
    print(selu_jit(1.0))

    1.05

Often you’ll see transformations applied using Python’s decorator syntax for convenience:

    @jax.jit
    def selu(x, alpha=1.67, lambda_=1.05):
      return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

## Tracing[\#](#tracing "Link to this heading")

The magic behind transformations is the notion of a [Tracer](glossary.html#term-Tracer). Tracers are abstract stand-ins for array objects, and are passed to JAX functions in order to extract the sequence of operations that the function encodes.

You can see this by printing any array value within transformed JAX code; for example:

    @jax.jit
    def f(x):
      print(x)
      return x + 1

    x = jnp.arange(5)
    result = f(x)

    JitTracer(int32[5])

The value printed is not the array `x`, but a `Tracer` instance that represents essential attributes of `x`, such as its `shape` and `dtype`. By executing the function with traced values, JAX can determine the sequence of operations encoded by the function before those operations are actually executed: transformations like [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"), and [`grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad") can then map this sequence of input operations to a transformed sequence of operations.

**Static vs traced operations**: Just as values can be either static or traced, operations can be static or traced. Static operations are evaluated at compile-time in Python; traced operations are compiled & evaluated at run-time in XLA.

For more details, see [Tracing](tracing.html#tracing-tutorial).

## Jaxprs[\#](#jaxprs "Link to this heading")

JAX has its own intermediate representation for sequences of operations, known as a [jaxpr](glossary.html#term-jaxpr). A jaxpr (short for *JAX exPRession*) is a simple representation of a functional program, comprising a sequence of [primitive](glossary.html#term-primitive) operations.

For example, consider the `selu` function we defined above:

    def selu(x, alpha=1.67, lambda_=1.05):
      return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

We can use the [`jax.make_jaxpr()`](_autosummary/jax.make_jaxpr.html#jax.make_jaxpr "jax.make_jaxpr") utility to convert this function into a jaxpr given a particular input:

    x = jnp.arange(5.0)
    jax.make_jaxpr(selu)(x)

    { lambda ; a:f32[5]. let
        b:bool[5] = gt a 0.0:f32[]
        c:f32[5] = exp a
        d:f32[5] = mul 1.6699999570846558:f32[] c
        e:f32[5] = sub d 1.6699999570846558:f32[]
        f:f32[5] = jit[
          name=_where
          jaxpr={ lambda ; b:bool[5] a:f32[5] e:f32[5]. let
              f:f32[5] = select_n b e a
            in (f,) }
        ] b a e
        g:f32[5] = mul 1.0499999523162842:f32[] f
      in (g,) }

Comparing this to the Python function definition, we see that it encodes the precise sequence of operations that the function represents. We’ll go into more depth about jaxprs later in [JAX internals: The jaxpr language](jaxpr.html#jax-internals-jaxpr).

## Pytrees[\#](#pytrees "Link to this heading")

JAX functions and transformations fundamentally operate on arrays, but in practice it is convenient to write code that works with collection of arrays: for example, a neural network might organize its parameters in a dictionary of arrays with meaningful keys. Rather than handle such structures on a case-by-case basis, JAX relies on the [pytree](glossary.html#term-pytree) abstraction to treat such collections in a uniform manner.

Here are some examples of objects that can be treated as pytrees:

    # (nested) list of parameters
    params = [1, 2, (jnp.arange(3), jnp.ones(2))]

    print(jax.tree.structure(params))
    print(jax.tree.leaves(params))

    PyTreeDef([*, *, (*, *)])
    [1, 2, Array([0, 1, 2], dtype=int32), Array([1., 1.], dtype=float32)]

    # Dictionary of parameters
    params = {'n': 5, 'W': jnp.ones((2, 2)), 'b': jnp.zeros(2)}

    print(jax.tree.structure(params))
    print(jax.tree.leaves(params))

    PyTreeDef({'W': *, 'b': *, 'n': *})
    [Array([[1., 1.],
           [1., 1.]], dtype=float32), Array([0., 0.], dtype=float32), 5]

    # Named tuple of parameters
    from typing import NamedTuple

    class Params(NamedTuple):
      a: int
      b: float

    params = Params(1, 5.0)
    print(jax.tree.structure(params))
    print(jax.tree.leaves(params))

    PyTreeDef(CustomNode(namedtuple[Params], [*, *]))
    [1, 5.0]

JAX has a number of general-purpose utilities for working with PyTrees; for example the functions [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") can be used to map a function to every leaf in a tree, and [`jax.tree.reduce()`](_autosummary/jax.tree.reduce.html#jax.tree.reduce "jax.tree.reduce") can be used to apply a reduction across the leaves in a tree.

You can learn more in the [Pytrees](pytrees.html#working-with-pytrees) tutorial.

## JAX API layering: NumPy, lax & XLA[\#](#jax-api-layering-numpy-lax-xla "Link to this heading")

All JAX operations are implemented in terms of operations in [XLA](https://www.openxla.org/xla/) – the Accelerated Linear Algebra compiler. If you look at the source of `jax.numpy`, you’ll see that all the operations are eventually expressed in terms of functions defined in [`jax.lax`](jax.lax.html#module-jax.lax "jax.lax"). While `jax.numpy` is a high-level wrapper that provides a familiar interface, you can think of `jax.lax` as a stricter, but often more powerful, lower-level API for working with multi-dimensional arrays.

For example, while `jax.numpy` will implicitly promote arguments to allow operations between mixed data types, `jax.lax` will not:

    import jax.numpy as jnp
    jnp.add(1, 1.0)  # jax.numpy API implicitly promotes mixed types.

    Array(2., dtype=float32, weak_type=True)

    from jax import lax
    lax.add(1, 1.0)  # jax.lax API requires explicit type promotion.

    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    Cell In[10], line 2
          1 from jax import lax
    ----> 2 lax.add(1, 1.0)  # jax.lax API requires explicit type promotion.

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/lax/lax.py:1164, in add(x, y)
       1144 r"""Elementwise addition: :math:`x + y`.
       1145 
       1146 This function lowers directly to the `stablehlo.add`_ operation.
       (...)   1161 .. _stablehlo.add: https://openxla.org/stablehlo/spec#add
       1162 """
       1163 x, y = core.auto_insert_reshard(x, y)
    -> 1164 return add_p.bind(x, y)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/core.py:689, in Primitive.bind(self, *args, **params)
        687 trace_ctx.set_trace(eval_trace)
        688 try:
    --> 689   return self.bind_with_trace(prev_trace, args, avals, params)
        690 finally:
        691   trace_ctx.set_trace(prev_trace)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/core.py:697, in Primitive.bind_with_trace(self, trace, args, avals, params)
        695   with set_current_trace(trace):
        696     return self.to_lojax(*args, **params)  # pyrefly: ignore[not-callable]
    --> 697 return trace.process_primitive(self, args, params)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/core.py:1278, in EvalTrace.process_primitive(self, primitive, args, params)
       1276 args = map(full_lower, args)
       1277 check_eval_args(args)
    -> 1278 return primitive.impl(*args, **params)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/dispatch.py:88, in apply_primitive(prim, *args, **params)
         86 prev = config.disable_jit.swap_local(False)
         87 try:
    ---> 88   outs = fun(*args)
         89 finally:
         90   config.disable_jit.set_local(prev)

        [... skipping hidden 15 frame]

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/lax/lax.py:9451, in check_same_dtypes(name, *avals)
       9449   equiv = _JNP_FUNCTION_EQUIVALENTS[name]
       9450   msg += f" (Tip: jnp.{equiv} is a similar function that does automatic type promotion on inputs)."
    -> 9451 raise TypeError(msg.format(name, ", ".join(str(a.dtype) for a in avals)))

    TypeError: lax.add requires arguments to have the same dtypes, got int32, float32. (Tip: jnp.add is a similar function that does automatic type promotion on inputs).

If using `jax.lax` directly, you’ll have to do type promotion explicitly in such cases:

    lax.add(jnp.float32(1), 1.0)

    Array(2., dtype=float32)

Along with this strictness, `jax.lax` also provides efficient APIs for some more general operations than are supported by NumPy.

For example, consider a 1D convolution, which can be expressed in NumPy this way:

    x = jnp.array([1, 2, 1])
    y = jnp.ones(10)
    jnp.convolve(x, y)

    Array([1., 3., 4., 4., 4., 4., 4., 4., 4., 4., 3., 1.], dtype=float32)

Under the hood, this NumPy operation is translated to a much more general convolution implemented by [`lax.conv_general_dilated`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html):

    from jax import lax
    result = lax.conv_general_dilated(
        x.reshape(1, 1, 3).astype(float),  # note: explicit promotion
        y.reshape(1, 1, 10),
        window_strides=(1,),
        padding=[(len(y) - 1, len(y) - 1)])  # equivalent of padding='full' in NumPy
    result[0, 0]

    Array([1., 3., 4., 4., 4., 4., 4., 4., 4., 4., 3., 1.], dtype=float32)

This is a batched convolution operation designed to be efficient for the types of convolutions often used in deep neural nets. It requires much more boilerplate, but is far more flexible and scalable than the convolution provided by NumPy (See [Convolutions in JAX](https://docs.jax.dev/en/latest/notebooks/convolutions.html) for more detail on JAX convolutions).

At their heart, all `jax.lax` operations are Python wrappers for operations in XLA; here, for example, the convolution implementation is provided by [XLA:ConvWithGeneralPadding](https://www.openxla.org/xla/operation_semantics#convwithgeneralpadding_convolution). Every JAX operation is eventually expressed in terms of these fundamental XLA operations, which is what enables just-in-time (JIT) compilation.

[](stateful-computations.html "previous page")

previous

Stateful computations

[](advanced_guides.html "next page")

next

Resources and Advanced Guides

Contents

- [Transformations](#transformations)
- [Tracing](#tracing)
- [Jaxprs](#jaxprs)
- [Pytrees](#pytrees)
- [JAX API layering: NumPy, lax & XLA](#jax-api-layering-numpy-lax-xla)

By The JAX authors

© Copyright 2024, The JAX Authors.\
