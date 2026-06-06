- [](index.html)
- [JAX 101](jax-101.html)
- Tracing

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/tracing.md "Download source file")
-  .pdf

# Tracing

## Contents

- [Static vs traced operations](#static-vs-traced-operations)
- [Different kinds of JAX values](#different-kinds-of-jax-values)

# Tracing[\#](#tracing "Link to this heading")

`jax.jit` and other JAX transforms work by *tracing* a function to determine its effect on inputs of a specific shape and type. For a window into tracing, let’s put a few `print()` statements within a JIT-compiled function and then call the function:

    from jax import jit
    import jax.numpy as jnp
    import numpy as np

    @jit
    def f(x, y):
      print("Running f():")
      print(f"  x = {x}")
      print(f"  y = {y}")
      result = jnp.dot(x + 1, y + 1)
      print(f"  result = {result}")
      return result

    x = np.random.randn(3, 4)
    y = np.random.randn(4)
    f(x, y)

    Running f():
      x = JitTracer(float32[3,4])
      y = JitTracer(float32[4])
      result = JitTracer(float32[3])

    Array([1.0172622, 4.61339  , 4.7813816], dtype=float32)

Notice that the print statements execute, but rather than printing the data we passed to the function, though, it prints *tracer* objects that stand-in for them (something like `Traced<ShapedArray(float32[])>`).

These tracer objects are what `jax.jit` uses to extract the sequence of operations specified by the function. Basic tracers are stand-ins that encode the **shape** and **dtype** of the arrays, but are agnostic to the values. This recorded sequence of computations can then be efficiently applied within XLA to new inputs with the same shape and dtype, without having to re-execute the Python code.

When we call the compiled function again on matching inputs, no re-compilation is required and nothing is printed because the result is computed in compiled XLA rather than in Python:

    x2 = np.random.randn(3, 4)
    y2 = np.random.randn(4)
    f(x2, y2)

    Array([4.4754734, 1.3044227, 4.792608 ], dtype=float32)

The extracted sequence of operations is encoded in a JAX expression, or [*jaxpr*](key-concepts.html#key-concepts-jaxprs) for short. You can view the jaxpr using the `jax.make_jaxpr` transformation:

    from jax import make_jaxpr

    def f(x, y):
      return jnp.dot(x + 1, y + 1)

    make_jaxpr(f)(x, y)

    { lambda ; a:f32[3,4] b:f32[4]. let
        c:f32[3,4] = add a 1.0:f32[]
        d:f32[4] = add b 1.0:f32[]
        e:f32[3] = dot_general[
          dimension_numbers=(([1], [0]), ([], []))
          preferred_element_type=float32
        ] c d
      in (e,) }

Note one consequence of this: because JIT compilation is done *without* information on the content of the array, control flow statements in the function cannot depend on traced values (see [Control flow and logical operators with JIT](control-flow.html#control-flow)). For example, this fails:

    @jit
    def f(x, neg):
      return -x if neg else x

    f(1, True)

    ---------------------------------------------------------------------------
    TracerBoolConversionError                 Traceback (most recent call last)
    Cell In[4], line 5
          1 @jit
          2 def f(x, neg):
          3   return -x if neg else x
          4 
    ----> 5 f(1, True)

        [... skipping hidden 5 frame]

    Cell In[4], line 3, in f(x, neg)
          1 @jit
          2 def f(x, neg):
    ----> 3   return -x if neg else x

        [... skipping hidden 1 frame]

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/core.py:1945, in concretization_function_error.<locals>.error(self, arg)
       1944 def error(self, arg):
    -> 1945   raise TracerBoolConversionError(arg)

    TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].
    The error occurred while tracing the function f at /tmp/ipykernel_3876/2422663986.py:1 for jit. This concrete value was not available in Python because it depends on the value of the argument neg.
    See https://docs.jax.dev/en/latest/errors.html#jax.errors.TracerBoolConversionError

If there are variables that you would not like to be traced, they can be marked as *static* for the purposes of JIT compilation:

    from functools import partial

    @jit(static_argnums=(1,))
    def f(x, neg):
      return -x if neg else x

    f(1, True)

    Array(-1, dtype=int32, weak_type=True)

Note that calling a JIT-compiled function with a different static argument results in re-compilation, so the function still works as expected:

    f(1, False)

    Array(1, dtype=int32, weak_type=True)

## Static vs traced operations[\#](#static-vs-traced-operations "Link to this heading")

Just as values can be either static or traced, operations can be static or traced. Static operations are evaluated at compile-time in Python; traced operations are compiled & evaluated at run-time in XLA.

This distinction between static and traced values makes it important to think about how to keep a static value static. Consider this function:

    import jax.numpy as jnp
    from jax import jit

    @jit
    def f(x):
      return x.reshape(jnp.array(x.shape).prod())

    x = jnp.ones((2, 3))
    f(x)

    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    Cell In[7], line 9
          5 def f(x):
          6   return x.reshape(jnp.array(x.shape).prod())
          7 
          8 x = jnp.ones((2, 3))
    ----> 9 f(x)

        [... skipping hidden 5 frame]

    Cell In[7], line 6, in f(x)
          4 @jit
          5 def f(x):
    ----> 6   return x.reshape(jnp.array(x.shape).prod())

        [... skipping hidden 2 frame]

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/numpy/array_methods.py:522, in _compute_newshape(arr, newshape)
        520 else:
        521   newshape: Sequence[DimSize]  # pyrefly: ignore[redefinition]
    --> 522 newshape = core.canonicalize_shape(newshape)
        523 neg1s = [i for i, d in enumerate(newshape) if type(d) is int and d == -1]
        524 if len(neg1s) > 1:

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/core.py:2119, in canonicalize_shape(shape, context)
       2117 except TypeError:
       2118   pass
    -> 2119 raise _invalid_shape_error(shape, context)

    TypeError: Shapes must be 1D sequences of concrete values of integer type, got [JitTracer(int32[])].
    If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
    The error occurred while tracing the function f at /tmp/ipykernel_3876/1983583872.py:4 for jit. This value became a tracer due to JAX operations on these lines:

      operation a:i32[] = reduce_prod[axes=(0,)] b
        from line /tmp/ipykernel_3876/1983583872.py:6:19 (f)

This fails with an error specifying that a tracer was found instead of a 1D sequence of concrete values of integer type. Let’s add some print statements to the function to understand why this is happening:

    @jit
    def f(x):
      print(f"x = {x}")
      print(f"x.shape = {x.shape}")
      print(f"jnp.array(x.shape).prod() = {jnp.array(x.shape).prod()}")
      # comment this out to avoid the error:
      # return x.reshape(jnp.array(x.shape).prod())

    f(x)

    x = JitTracer(float32[2,3])
    x.shape = (2, 3)
    jnp.array(x.shape).prod() = JitTracer(int32[])

Notice that although `x` is traced, `x.shape` is a static value. However, when we use `jnp.array` and `jnp.prod` on this static value, it becomes a traced value, at which point it cannot be used in a function like `reshape()` that requires a static input (recall: array shapes must be static).

A useful pattern is to use `numpy` for operations that should be static (i.e. done at compile-time), and use `jax.numpy` for operations that should be traced (i.e. compiled and executed at run-time). For this function, it might look like this:

    from jax import jit
    import jax.numpy as jnp
    import numpy as np

    @jit
    def f(x):
      return x.reshape((np.prod(x.shape),))

    f(x)

    Array([1., 1., 1., 1., 1., 1.], dtype=float32)

For this reason, a standard convention in JAX programs is to `import`` ``numpy`` ``as`` ``np` and `import`` ``jax.numpy`` ``as`` ``jnp` so that both interfaces are available for finer control over whether operations are performed in a static manner (with `numpy`, once at compile-time) or a traced manner (with `jax.numpy`, optimized at run-time).

Understanding which values and operations will be static and which will be traced is a key part of using `jax.jit` effectively.

## Different kinds of JAX values[\#](#different-kinds-of-jax-values "Link to this heading")

A tracer value carries an **abstract** value, e.g., `ShapedArray` with information about the shape and dtype of an array. We will refer here to such tracers as **abstract tracers**. Some tracers, e.g., those that are introduced for arguments of autodiff transformations, carry `ConcreteArray` abstract values that actually include the regular array data, and are used, e.g., for resolving conditionals. We will refer here to such tracers as **concrete tracers**. Tracer values computed from these concrete tracers, perhaps in combination with regular values, result in concrete tracers. A **concrete value** is either a regular value or a concrete tracer.

Typically, computations that involve at least a tracer value will produce a tracer value. There are very few exceptions, when a computation can be entirely done using the abstract value carried by a tracer, in which case the result can be a **regular** Python value. For example, getting the shape of a tracer with `ShapedArray` abstract value. Another example is when explicitly casting a concrete tracer value to a regular type, e.g., `int(x)` or `x.astype(float)`. Another such situation is for `bool(x)`, which produces a Python bool when concreteness makes it possible. That case is especially salient because of how often it arises in control flow.

Here is how the transformations introduce abstract or concrete tracers:

- [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"): introduces **abstract tracers** for all positional arguments except those denoted by `static_argnums`, which remain regular values.

- [`jax.pmap()`](_autosummary/jax.pmap.html#jax.pmap "jax.pmap"): introduces **abstract tracers** for all positional arguments except those denoted by `static_broadcasted_argnums`.

- [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"), [`jax.make_jaxpr()`](_autosummary/jax.make_jaxpr.html#jax.make_jaxpr "jax.make_jaxpr"), `xla_computation()`: introduce **abstract tracers** for all positional arguments.

- [`jax.jvp()`](_autosummary/jax.jvp.html#jax.jvp "jax.jvp") and [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad") introduce **concrete tracers** for all positional arguments. An exception is when these transformations are within an outer transformation and the actual arguments are themselves abstract tracers; in that case, the tracers introduced by the autodiff transformations are also abstract tracers.

- All higher-order control-flow primitives (`lax.cond()`, `lax.while_loop()`, `lax.fori_loop()`, `lax.scan()`) when they process the functionals introduce **abstract tracers**, whether or not there is a JAX transformation in progress.

All of this is relevant when you have code that can operate only on regular Python values, such as code that has conditional control-flow based on data:

    def divide(x, y):
      return x / y if y >= 1. else 0.

If we want to apply [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), we must ensure to specify `static_argnums=1` to ensure `y` stays a regular value. This is due to the boolean expression `y`` ``>=`` ``1.`, which requires concrete values (regular or tracers). The same would happen if we write explicitly `bool(y`` ``>=`` ``1.)`, or `int(y)`, or `float(y)`.

Interestingly, `jax.grad(divide)(3.,`` ``2.)`, works because [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad") uses concrete tracers, and resolves the conditional using the concrete value of `y`.

[](control-flow.html "previous page")

previous

Control flow and logical operators with JIT

[](stateful-computations.html "next page")

next

Stateful computations

Contents

- [Static vs traced operations](#static-vs-traced-operations)
- [Different kinds of JAX values](#different-kinds-of-jax-values)

By The JAX authors

© Copyright 2024, The JAX Authors.\
