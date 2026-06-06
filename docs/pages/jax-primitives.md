- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- JAX Internals: primitives

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/jax-primitives.md "Download source file")
-  .pdf

# JAX Internals: primitives

## Contents

- [Introduction to JAX primitives](#introduction-to-jax-primitives)
- [Using existing JAX primitives](#using-existing-jax-primitives)
- [Defining new JAX primitives](#defining-new-jax-primitives)
  - [Primal evaluation rules](#primal-evaluation-rules)
  - [What happens when you use `jit`](#what-happens-when-you-use-jit)
    - [Abstract evaluation rules](#abstract-evaluation-rules)
    - [XLA Compilation rules](#xla-compilation-rules)
  - [Forward differentiation](#forward-differentiation)
    - [JIT of forward differentiation](#jit-of-forward-differentiation)
  - [Reverse differentiation](#reverse-differentiation)
    - [Transposition](#transposition)
    - [JIT of reverse differentiation](#jit-of-reverse-differentiation)
  - [Batching](#batching)
    - [JIT of batching](#jit-of-batching)

# JAX Internals: primitives[\#](#jax-internals-primitives "Link to this heading")

## Introduction to JAX primitives[\#](#introduction-to-jax-primitives "Link to this heading")

A JAX primitive is the basic computational unit of a JAX program. This document explains the interface that a JAX primitive must support to allow JAX to perform all its transformations (this is not a how-to guide).

For example, the multiply-add operation can be implemented in terms of the low-level `jax.lax.*` primitives (which are like XLA operator wrappers) or `jax.extend.core.Primitive("multiply_add")`, as demonstrated further below.

And JAX is able to take sequences of such primitive operations, and transform them via its composable transformations of Python functions, such as [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad") and [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"). JAX implements these transforms in a *JAX-traceable* way. This means that when a Python function is executed, the only operations it applies to the data are either:

- **Inspections of data attributes:** Data information, such as shape or type; or

- **JAX primitives:** These are the JAX special operations covered in this tutorial.

JAX primitives know how to operate on both concrete data values and abstract JAX values. *A JAX-traceable function* can be invoked by JAX with abstract arguments. For example, a JAX abstract value — `ShapedArray(float32[2,2])` — captures the type and the shape of values, but not the concrete data values.

The JAX-transformed functions must themselves be JAX-traceable functions *to make sure that these transformations are composable*, for example like `jax.jit(jax.jacfwd(jax.grad(f)))`.

JAX provides pre-defined primitives corresponding to most XLA operations, including add, matmul, sin, cos, and indexing.

In addition, JAX offers an implementation of NumPy functions in terms of JAX primitives. This means that *Python programs using JAX’s implementation of NumPy are JAX-traceable and, therefore, transformable*. Other libraries can be made JAX-traceable by implementing them in terms of JAX primitives.

Furthermore, the set of JAX primitives is extensible, so instead of reimplementing a function in terms of pre-defined JAX primitives, you can define a new primitive that encapsulates the behavior of the function.

Consider the following example: you want to add to JAX support for a multiply-add function with three arguments, defined mathematically as `multiply_add(x,`` ``y,`` ``z)`` ``=`` ``x`` ``*`` ``y`` ``+`` ``z`. This function operates on 3 identically-shaped tensors of floating point values and performs the operations pointwise. You can do this by:

- [Using existing JAX primitives](#using-existing-jax-primitives); or

- [Defining new JAX primitives](#defining-new-jax-primitives)

## Using existing JAX primitives[\#](#using-existing-jax-primitives "Link to this heading")

The easiest way to define new functions is to write them in terms of JAX primitives, or in terms of other functions that are themselves written using JAX primitives, for example, those defined in the [`jax.lax()`](jax.lax.html#module-jax.lax "jax.lax") module:

    from jax._src.lax import lax
    from jax._src import api

    def multiply_add_lax(x, y, z):
      """Implementation of multiply-add using the `jax.lax` primitives."""
      return lax.add(lax.mul(x, y), z)


    def square_add_lax(a, b):
      """A square-add function using the newly defined multiply-add."""
      return multiply_add_lax(a, a, b)

    print("square_add_lax = ", square_add_lax(2., 10.))
    # Differentiate w.r.t. the first argument
    print("grad(square_add_lax) = ", api.grad(square_add_lax, argnums=0)(2.0, 10.))

    square_add_lax =  14.0
    grad(square_add_lax) =  4.0

To understand how JAX is internally using the primitives, add some helpers for tracing function calls:

    #@title Helper functions (execute this cell)
    import functools
    import traceback

    _indentation = 0
    def _trace(msg=None):
        """Print a message at current indentation."""
        if msg is not None:
            print("  " * _indentation + msg)

    def _trace_indent(msg=None):
        """Print a message and then indent the rest."""
        global _indentation
        _trace(msg)
        _indentation = 1 + _indentation

    def _trace_unindent(msg=None):
        """Unindent then print a message."""
        global _indentation
        _indentation = _indentation - 1
        _trace(msg)

    def trace(name):
      """A decorator for functions to trace arguments and results."""

      def trace_func(func):
        def pp(v):
            """Print certain values more succinctly"""
            vtype = str(type(v))
            if "jax._src.xla_bridge._JaxComputationBuilder" in vtype:
                return "<JaxComputationBuilder>"
            elif "jaxlib._jax_.XlaOp" in vtype:
                return "<XlaOp at 0x{:x}>".format(id(v))
            elif ("partial_eval.JaxprTracer" in vtype or
                  "batching.BatchTracer" in vtype or
                  "ad.JVPTracer" in vtype):
                return "Traced<{}>".format(v.aval)
            elif isinstance(v, tuple):
                return "({})".format(pp_values(v))
            else:
                return str(v)
        def pp_values(args):
            return ", ".join([pp(arg) for arg in args])

        @functools.wraps(func)
        def func_wrapper(*args):
          _trace_indent("call {}({})".format(name, pp_values(args)))
          res = func(*args)
          _trace_unindent("|<- {} = {}".format(name, pp(res)))
          return res

        return func_wrapper

      return trace_func

    class expectNotImplementedError(object):
      """Context manager to check for NotImplementedError."""
      def __enter__(self): pass
      def __exit__(self, type, value, tb):
        global _indentation
        _indentation = 0
        if type is NotImplementedError:
          print("\nFound expected exception:")
          traceback.print_exc(limit=3)
          return True
        elif type is None:  # No exception
          assert False, "Expected NotImplementedError"
        else:
          return False

Instead of using [`jax.lax()`](jax.lax.html#module-jax.lax "jax.lax") primitives directly, you can use other functions that are already written in terms of those primitives, such as those in `jax.numpy`:

    import jax.numpy as jnp
    import numpy as np

    @trace("multiply_add_numpy")
    def multiply_add_numpy(x, y, z):
        return jnp.add(jnp.multiply(x, y), z)

    @trace("square_add_numpy")
    def square_add_numpy(a, b):
        return multiply_add_numpy(a, a, b)

    print("\nNormal evaluation:")
    print("square_add_numpy = ", square_add_numpy(2., 10.))
    print("\nGradient evaluation:")
    print("grad(square_add_numpy) = ", api.grad(square_add_numpy)(2.0, 10.))

    Normal evaluation:
    call square_add_numpy(2.0, 10.0)
      call multiply_add_numpy(2.0, 2.0, 10.0)
      |<- multiply_add_numpy = 14.0
    |<- square_add_numpy = 14.0
    square_add_numpy =  14.0

    Gradient evaluation:
    call square_add_numpy(GradTracer(primal=2.0, typeof(tangent)=f32[]), 10.0)
      call multiply_add_numpy(GradTracer(primal=2.0, typeof(tangent)=f32[]), GradTracer(primal=2.0, typeof(tangent)=f32[]), 10.0)
      |<- multiply_add_numpy = GradTracer(primal=14.0, typeof(tangent)=f32[])
    |<- square_add_numpy = GradTracer(primal=14.0, typeof(tangent)=f32[])
    grad(square_add_numpy) =  4.0

Notice that in the process of computing [`jax.grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"), JAX invokes `square_add_numpy` and `multiply_add_numpy` with special arguments `ConcreteArray(...)` (described further below in this colab). It is important to remember that a JAX-traceable function must be able to operate not only on concrete arguments but also on special abstract arguments that JAX may use to abstract the function execution.

The JAX traceability property is satisfied as long as the function is written in terms of JAX primitives.

## Defining new JAX primitives[\#](#defining-new-jax-primitives "Link to this heading")

The right way to add support for multiply-add is in terms of existing JAX primitives, as shown above. However, to demonstrate how JAX primitives work, pretend that you want to add a new primitive to JAX for the multiply-add functionality.

    from jax.extend import core

    multiply_add_p = core.Primitive("multiply_add")  # Create the primitive

    @trace("multiply_add_prim")
    def multiply_add_prim(x, y, z):
      """The JAX-traceable way to use the JAX primitive.

      Note that the traced arguments must be passed as positional arguments
      to `bind`.
      """
      return multiply_add_p.bind(x, y, z)

    @trace("square_add_prim")
    def square_add_prim(a, b):
      """A square-add function implemented using the new JAX-primitive."""
      return multiply_add_prim(a, a, b)

If you try to call the newly defined functions, you’ll get an error, because you haven’t yet told JAX anything about the semantics of the new primitive.

    with expectNotImplementedError():
      square_add_prim(2., 10.)

    call square_add_prim(2.0, 10.0)
      call multiply_add_prim(2.0, 2.0, 10.0)

    Found expected exception:

    Traceback (most recent call last):
      File "/tmp/ipykernel_2055/2844449444.py", line 2, in <module>
        square_add_prim(2., 10.)
      File "/tmp/ipykernel_2055/3025987661.py", line 48, in func_wrapper
        res = func(*args)
              ^^^^^^^^^^^
      File "/tmp/ipykernel_2055/3275395289.py", line 17, in square_add_prim
        return multiply_add_prim(a, a, b)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^
    NotImplementedError: Evaluation rule for 'multiply_add' not implemented

### Primal evaluation rules[\#](#primal-evaluation-rules "Link to this heading")

    @trace("multiply_add_impl")
    def multiply_add_impl(x, y, z):
      """Concrete implementation of the primitive.

      This function does not need to be JAX traceable.

      Args:
        x, y, z: The concrete arguments of the primitive. Will only be called with
          concrete values.

      Returns:
        the concrete result of the primitive.
      """
      # Note: you can use the ordinary (non-JAX) NumPy, which is not JAX-traceable.
      return np.add(np.multiply(x, y), z)

    # Now, register the primal implementation with JAX:
    multiply_add_p.def_impl(multiply_add_impl)

    <function __main__.multiply_add_impl(x, y, z)>

    assert square_add_prim(2., 10.) == 14.

    call square_add_prim(2.0, 10.0)
      call multiply_add_prim(2.0, 2.0, 10.0)
        call multiply_add_impl(2.0, 2.0, 10.0)
        |<- multiply_add_impl = 14.0
      |<- multiply_add_prim = 14.0
    |<- square_add_prim = 14.0

### What happens when you use `jit`[\#](#what-happens-when-you-use-jit "Link to this heading")

Now, if you try to use `jit`, you’ll get a `NotImplementedError`:

    with expectNotImplementedError():
      api.jit(square_add_prim)(2., 10.)

    call square_add_prim(JitTracer(~float32[]), JitTracer(~float32[]))
      call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[]))

    Found expected exception:

    Traceback (most recent call last):
      File "/tmp/ipykernel_2055/1813425700.py", line 2, in <module>
        api.jit(square_add_prim)(2., 10.)
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/traceback_util.py", line 195, in reraise_with_filtered_traceback
        return fun(*args, **kwargs)  # pyrefly: ignore[not-callable]
               ^^^^^^^^^^^^^^^^^^^^
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/pjit.py", line 255, in cache_miss
        p, args_flat = _infer_params(fun, jit_info, args, kwargs)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    NotImplementedError: Abstract evaluation for 'multiply_add' not implemented

#### Abstract evaluation rules[\#](#abstract-evaluation-rules "Link to this heading")

To JIT the function, and for other transformations as well, JAX first evaluates it abstractly using only the shape and type of the arguments. This abstract evaluation serves multiple purposes:

- Gets the sequence of JAX primitives that are used in the computation. This sequence will be compiled.

- Computes the shape and type of all vectors and operations used in the computation.

For example, the abstraction of a vector with 3 elements may be `ShapedArray(float32[3])`, or `ConcreteArray([1.,`` ``2.,`` ``3.])`. In the latter case, JAX uses the actual concrete value wrapped as an abstract value.

    from jax import core

    @trace("multiply_add_abstract_eval")
    def multiply_add_abstract_eval(xs, ys, zs):
      """Abstract evaluation of the primitive.

      This function does not need to be JAX traceable. It will be invoked with
      abstractions of the actual arguments

      Args:
        xs, ys, zs: Abstractions of the arguments.

      Result:
        a ShapedArray for the result of the primitive.
      """
      assert xs.shape == ys.shape
      assert xs.shape == zs.shape
      return core.ShapedArray(xs.shape, xs.dtype)

    # Now, register the abstract evaluation with JAX:
    multiply_add_p.def_abstract_eval(multiply_add_abstract_eval)

    <function __main__.multiply_add_abstract_eval(xs, ys, zs)>

If you re-attempt to apply `jit`, you can inspect how the abstract evaluation proceeds, but you’ll get another error about missing the actual XLA compilation rule:

    with expectNotImplementedError():
      api.jit(square_add_prim)(2., 10.)

    call square_add_prim(JitTracer(~float32[]), JitTracer(~float32[]))
      call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[]))
        call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
        |<- multiply_add_abstract_eval = float32[]
      |<- multiply_add_prim = JitTracer(float32[])
    |<- square_add_prim = JitTracer(float32[])

    Found expected exception:

    Traceback (most recent call last):
      File "/tmp/ipykernel_2055/3025987661.py", line 48, in func_wrapper
        res = func(*args)
      File "/tmp/ipykernel_2055/3275395289.py", line 17, in square_add_prim
        return multiply_add_prim(a, a, b)
      File "/tmp/ipykernel_2055/3025987661.py", line 48, in func_wrapper
        res = func(*args)
    jax._src.source_info_util.JaxStackTraceBeforeTransformation: NotImplementedError: MLIR translation rule for primitive 'multiply_add' not found for platform cpu

    The preceding stack trace is the source of the JAX operation that, once transformed by JAX, triggered the following exception.

    --------------------

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/tmp/ipykernel_2055/1813425700.py", line 2, in <module>
        api.jit(square_add_prim)(2., 10.)
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/traceback_util.py", line 195, in reraise_with_filtered_traceback
        return fun(*args, **kwargs)  # pyrefly: ignore[not-callable]
               ^^^^^^^^^^^^^^^^^^^^
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/pjit.py", line 257, in cache_miss
        executable, pgle_profiler, const_args) = _run_python_pjit(
                                                 ^^^^^^^^^^^^^^^^^
    NotImplementedError: MLIR translation rule for primitive 'multiply_add' not found for platform cpu

#### XLA Compilation rules[\#](#xla-compilation-rules "Link to this heading")

JAX compilation works by compiling each primitive into a graph of XLA operations.

This is the biggest hurdle to adding new functionality to JAX, because the set of XLA operations is limited, and JAX already has pre-defined primitives for most of them. However, XLA includes a `CustomCall` operation that can be used to encapsulate arbitrary functionality defined using C++.

    from jax._src.lib.mlir.dialects import hlo

    @trace("multiply_add_lowering")
    def multiply_add_lowering(ctx, xc, yc, zc):
      """The compilation to XLA of the primitive.

      Given an mlir.ir.Value for each argument, return the mlir.ir.Values for
      the results of the function.

      Does not need to be a JAX-traceable function.
      """
      return [hlo.AddOp(hlo.MulOp(xc, yc), zc).result]

    # Now, register the lowering rule with JAX.
    # For GPU, refer to the https://docs.jax.dev/en/latest/Custom_Operation_for_GPUs.html
    from jax.interpreters import mlir

    mlir.register_lowering(multiply_add_p, multiply_add_lowering, platform='cpu')

You will now succeed to apply `jax.jit`. Notice below that JAX first evaluates the function abstractly, which triggers the `multiply_add_abstract_eval` function, and then compiles the set of primitives it has encountered, including `multiply_add`. At this point JAX invokes `multiply_add_lowering`.

    assert api.jit(lambda x, y: square_add_prim(x, y))(2., 10.) == 14.

    call square_add_prim(JitTracer(~float32[]), JitTracer(~float32[]))
      call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[]))
        call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
        |<- multiply_add_abstract_eval = float32[]
      |<- multiply_add_prim = JitTracer(float32[])
    |<- square_add_prim = JitTracer(float32[])
    call multiply_add_lowering(LoweringRuleContext(module_context=ModuleContext(context=<jax._src.interpreters.mlir.JaxIrContext object at 0x7e02edabae50>, module=<jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7e02edabf150>, ip=<jaxlib.mlir._mlir_libs._mlir.ir.InsertionPoint object at 0x7e02edabf190>, symbol_table=<jaxlib.mlir._mlir_libs._mlir.ir.SymbolTable object at 0x7e02f40efae0>, platforms=('cpu',), backend=<jaxlib._jax.Client object at 0x7e02f5ebe740>, axis_context=ShardingContext(num_devices=1, device_assignment=None, abstract_mesh=None), keepalives=[], channel_iterator=count(2), host_callbacks=[], shape_poly_state=<jax._src.interpreters.mlir.ShapePolyLoweringState object at 0x7e02eda65a00>, all_default_mem_kind=True, lowering_cache={}, cached_primitive_lowerings={}, sharding_attr_cache={}, aval_to_ir_types_cache={ShapedArray(float32[], weak_type=True): RankedTensorType(tensor<f32>), ShapedArray(float32[]): RankedTensorType(tensor<f32>), ShapedArray(int32[]): RankedTensorType(tensor<i32>)}, pallas_lowering_cache={}, traceback_caches=TracebackCaches(traceback_to_location_cache=<jaxlib.mlir._mlir_libs._jax_mlir_ext.TracebackToLocationCache object at 0x7e02eda813e0>, canonical_name_cache={}), lowering_parameters=LoweringParameters(override_lowering_rules=None, global_constant_computation=False, for_export=False, export_ignore_forward_compatibility=False, hoist_constants_as_args=False)), name_stack=NameStack(stack=()), traceback=None, primitive=multiply_add, avals_in=(ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True)), avals_out=[ShapedArray(float32[])], tokens_in=<jax._src.interpreters.mlir.TokenSet object at 0x7e02eda65f30>, tokens_out=None, const_lowering={}, axis_size_env=None, dim_var_values=[], jaxpr_eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), platforms=None), BlockArgument(<block argument> of type 'tensor<f32>' at index: 0), BlockArgument(<block argument> of type 'tensor<f32>' at index: 1), BlockArgument(<block argument> of type 'tensor<f32>' at index: 2))
    |<- multiply_add_lowering = [<jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x7e02f40bb070>]

Below is another use of `jit`, where you compile only with respect to the first argument. Notice how the second argument to `square_add_prim` is concrete, which leads in the third argument to `multiply_add_abstract_eval` being `ConcreteArray`. Notice that `multiply_add_abstract_eval` may be used with both `ShapedArray` and `ConcreteArray`.

    assert api.jit(lambda x, y: square_add_prim(x, y),
                   static_argnums=1)(2., 10.) == 14.

    call square_add_prim(JitTracer(~float32[]), 10.0)
      call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), 10.0)
        call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
        |<- multiply_add_abstract_eval = float32[]
      |<- multiply_add_prim = JitTracer(float32[])
    |<- square_add_prim = JitTracer(float32[])
    call multiply_add_lowering(LoweringRuleContext(module_context=ModuleContext(context=<jax._src.interpreters.mlir.JaxIrContext object at 0x7e02edabb6d0>, module=<jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7e02edabfe20>, ip=<jaxlib.mlir._mlir_libs._mlir.ir.InsertionPoint object at 0x7e02edabfe60>, symbol_table=<jaxlib.mlir._mlir_libs._mlir.ir.SymbolTable object at 0x7e02eda66c70>, platforms=('cpu',), backend=<jaxlib._jax.Client object at 0x7e02f5ebe740>, axis_context=ShardingContext(num_devices=1, device_assignment=None, abstract_mesh=None), keepalives=[], channel_iterator=count(2), host_callbacks=[], shape_poly_state=<jax._src.interpreters.mlir.ShapePolyLoweringState object at 0x7e02eda66cc0>, all_default_mem_kind=True, lowering_cache={}, cached_primitive_lowerings={}, sharding_attr_cache={}, aval_to_ir_types_cache={ShapedArray(float32[], weak_type=True): RankedTensorType(tensor<f32>), ShapedArray(float32[]): RankedTensorType(tensor<f32>), ShapedArray(int32[]): RankedTensorType(tensor<i32>)}, pallas_lowering_cache={}, traceback_caches=TracebackCaches(traceback_to_location_cache=<jaxlib.mlir._mlir_libs._jax_mlir_ext.TracebackToLocationCache object at 0x7e02eda81470>, canonical_name_cache={}), lowering_parameters=LoweringParameters(override_lowering_rules=None, global_constant_computation=False, for_export=False, export_ignore_forward_compatibility=False, hoist_constants_as_args=False)), name_stack=NameStack(stack=()), traceback=None, primitive=multiply_add, avals_in=(ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True)), avals_out=[ShapedArray(float32[])], tokens_in=<jax._src.interpreters.mlir.TokenSet object at 0x7e02eda673d0>, tokens_out=None, const_lowering={}, axis_size_env=None, dim_var_values=[], jaxpr_eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), platforms=None), BlockArgument(<block argument> of type 'tensor<f32>' at index: 0), BlockArgument(<block argument> of type 'tensor<f32>' at index: 1), BlockArgument(<block argument> of type 'tensor<f32>' at index: 2))
    |<- multiply_add_lowering = [<jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x7e02edad6af0>]

### Forward differentiation[\#](#forward-differentiation "Link to this heading")

JAX implements forward differentiation in the form of a Jacobian-Vector Product (JVP) (you can learn more about it in [Forward- and reverse-mode autodiff in JAX](jacobian-vector-products.html#advanced-guides-jvp-vjp)).

If you attempt to compute the `jvp` function, you’ll get an error because you have not yet told JAX how to differentiate the `multiply_add` primitive.

    # The second argument is set to `(2., 10.)` values where you
    # evaluate the Jacobian, and the third argument `(1., 1.)`
    # contains the values of the tangents for the arguments.
    with expectNotImplementedError():
      api.jvp(square_add_prim, (2., 10.), (1., 1.))

    call square_add_prim(Traced<~float32[]>, Traced<~float32[]>)
      call multiply_add_prim(Traced<~float32[]>, Traced<~float32[]>, Traced<~float32[]>)

    Found expected exception:

    Traceback (most recent call last):
      File "/tmp/ipykernel_2055/459539105.py", line 5, in <module>
        api.jvp(square_add_prim, (2., 10.), (1., 1.))
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/traceback_util.py", line 195, in reraise_with_filtered_traceback
        return fun(*args, **kwargs)  # pyrefly: ignore[not-callable]
               ^^^^^^^^^^^^^^^^^^^^
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/api.py", line 1390, in jvp
        return _jvp(fun, primals, tangents, has_aux=has_aux)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    NotImplementedError: Differentiation rule for 'multiply_add' not implemented

    from jax.interpreters import ad

    @trace("multiply_add_value_and_jvp")
    def multiply_add_value_and_jvp(arg_values, arg_tangents):
      """Evaluates the primal output and the tangents (Jacobian-vector product).

      Given values of the arguments and perturbation of the arguments (tangents),
      compute the output of the primitive and the perturbation of the output.

      This method must be JAX-traceable. JAX may invoke it with abstract values
      for the arguments and tangents.

      Args:
        arg_values: A tuple of arguments
        arg_tangents: A tuple with the tangents of the arguments. The tuple has
          the same length as the arg_values. Some of the tangents may also be the
          special value `ad.Zero` to specify a zero tangent

      Returns:
         A pair of the primal output and the tangent.
      """
      x, y, z = arg_values
      xt, yt, zt = arg_tangents
      _trace("Primal evaluation:")
      # Now, you have a JAX-traceable computation of the output.
      # Normally, you can use the multiply add (`ma`) primitive itself to compute the primal output.
      primal_out = multiply_add_prim(x, y, z)

      _trace("Tangent evaluation:")
      # You must use a JAX-traceable way to compute the tangent. It turns out that
      # the output tangent can be computed as (xt * y + x * yt + zt),
      # which you can implement in a JAX-traceable way using the same "multiply_add_prim" primitive.

      # You do need to deal specially with `Zero`. Here, you just turn it into a
      # proper tensor of 0s (of the same shape as 'x').
      # An alternative would be to check for `Zero` and perform algebraic
      # simplification of the output tangent computation.
      def make_zero(tan):
        return lax.full_like(x, 0) if type(tan) is ad.Zero else tan

      output_tangent = multiply_add_prim(make_zero(xt), y, multiply_add_prim(x, make_zero(yt), make_zero(zt)))
      return (primal_out, output_tangent)

    # Register the forward differentiation rule with JAX:
    ad.primitive_jvps[multiply_add_p] = multiply_add_value_and_jvp

    # Tangent is: xt*y + x*yt + zt = 1.*2. + 2.*1. + 1. = 5.
    assert api.jvp(square_add_prim, (2., 10.), (1., 1.)) == (14., 5.)

    call square_add_prim(Traced<~float32[]>, Traced<~float32[]>)
      call multiply_add_prim(Traced<~float32[]>, Traced<~float32[]>, Traced<~float32[]>)
        call multiply_add_value_and_jvp((2.0, 2.0, 10.0), (1.0, 1.0, 1.0))
          Primal evaluation:
          call multiply_add_prim(2.0, 2.0, 10.0)
            call multiply_add_impl(2.0, 2.0, 10.0)
            |<- multiply_add_impl = 14.0
          |<- multiply_add_prim = 14.0
          Tangent evaluation:
          call multiply_add_prim(2.0, 1.0, 1.0)
            call multiply_add_impl(2.0, 1.0, 1.0)
            |<- multiply_add_impl = 3.0
          |<- multiply_add_prim = 3.0
          call multiply_add_prim(1.0, 2.0, 3.0)
            call multiply_add_impl(1.0, 2.0, 3.0)
            |<- multiply_add_impl = 5.0
          |<- multiply_add_prim = 5.0
        |<- multiply_add_value_and_jvp = (14.0, 5.0)
      |<- multiply_add_prim = Traced<float32[]>
    |<- square_add_prim = Traced<float32[]>

#### JIT of forward differentiation[\#](#jit-of-forward-differentiation "Link to this heading")

You can apply `jit` to the forward differentiation function:

    assert api.jit(lambda arg_values, arg_tangents:
                       api.jvp(square_add_prim, arg_values, arg_tangents))(
             (2., 10.), (1., 1.)) == (14., 5.)

    call square_add_prim(Traced<~float32[]>, Traced<~float32[]>)
      call multiply_add_prim(Traced<~float32[]>, Traced<~float32[]>, Traced<~float32[]>)
        call multiply_add_value_and_jvp((JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[])), (JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[])))
          Primal evaluation:
          call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[]))
            call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = JitTracer(float32[])
          Tangent evaluation:
          call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[]))
            call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = JitTracer(float32[])
          call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(float32[]))
            call multiply_add_abstract_eval(~float32[], ~float32[], float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = JitTracer(float32[])
        |<- multiply_add_value_and_jvp = (JitTracer(float32[]), JitTracer(float32[]))
      |<- multiply_add_prim = Traced<float32[]>
    |<- square_add_prim = Traced<float32[]>
    call multiply_add_lowering(LoweringRuleContext(module_context=ModuleContext(context=<jax._src.interpreters.mlir.JaxIrContext object at 0x7e02ed90f1d0>, module=<jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7e02ed920cc0>, ip=<jaxlib.mlir._mlir_libs._mlir.ir.InsertionPoint object at 0x7e02ed920d00>, symbol_table=<jaxlib.mlir._mlir_libs._mlir.ir.SymbolTable object at 0x7e02edb0fae0>, platforms=('cpu',), backend=<jaxlib._jax.Client object at 0x7e02f5ebe740>, axis_context=ShardingContext(num_devices=1, device_assignment=None, abstract_mesh=None), keepalives=[], channel_iterator=count(2), host_callbacks=[], shape_poly_state=<jax._src.interpreters.mlir.ShapePolyLoweringState object at 0x7e02ed9147a0>, all_default_mem_kind=True, lowering_cache={}, cached_primitive_lowerings={}, sharding_attr_cache={}, aval_to_ir_types_cache={ShapedArray(float32[], weak_type=True): RankedTensorType(tensor<f32>), ShapedArray(float32[]): RankedTensorType(tensor<f32>), ShapedArray(int32[]): RankedTensorType(tensor<i32>)}, pallas_lowering_cache={}, traceback_caches=TracebackCaches(traceback_to_location_cache=<jaxlib.mlir._mlir_libs._jax_mlir_ext.TracebackToLocationCache object at 0x7e02eda833f0>, canonical_name_cache={}), lowering_parameters=LoweringParameters(override_lowering_rules=None, global_constant_computation=False, for_export=False, export_ignore_forward_compatibility=False, hoist_constants_as_args=False)), name_stack=NameStack(stack=()), traceback=None, primitive=multiply_add, avals_in=(ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True)), avals_out=[ShapedArray(float32[])], tokens_in=<jax._src.interpreters.mlir.TokenSet object at 0x7e02ed915000>, tokens_out=None, const_lowering={}, axis_size_env=None, dim_var_values=[], jaxpr_eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), platforms=None), BlockArgument(<block argument> of type 'tensor<f32>' at index: 0), BlockArgument(<block argument> of type 'tensor<f32>' at index: 1), BlockArgument(<block argument> of type 'tensor<f32>' at index: 2))
    |<- multiply_add_lowering = [<jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x7e02edad9470>]
    call multiply_add_lowering(LoweringRuleContext(module_context=ModuleContext(context=<jax._src.interpreters.mlir.JaxIrContext object at 0x7e02ed90f1d0>, module=<jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7e02ed920cc0>, ip=<jaxlib.mlir._mlir_libs._mlir.ir.InsertionPoint object at 0x7e02ed920d00>, symbol_table=<jaxlib.mlir._mlir_libs._mlir.ir.SymbolTable object at 0x7e02edb0fae0>, platforms=('cpu',), backend=<jaxlib._jax.Client object at 0x7e02f5ebe740>, axis_context=ShardingContext(num_devices=1, device_assignment=None, abstract_mesh=None), keepalives=[], channel_iterator=count(2), host_callbacks=[], shape_poly_state=<jax._src.interpreters.mlir.ShapePolyLoweringState object at 0x7e02ed9147a0>, all_default_mem_kind=True, lowering_cache={LoweringCacheKey(primitive=multiply_add, eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), avals_in=(ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True)), effects=frozenset(), params=(), platforms=('cpu',)): LoweringCacheValue(func=<jaxlib.mlir.dialects.func.FuncOp object at 0x7e02ed921040>, flat_output_types=[RankedTensorType(tensor<f32>)], output_treedef=PyTreeDef([*]), const_args=(), const_arg_avals=(), inline=True)}, cached_primitive_lowerings={}, sharding_attr_cache={}, aval_to_ir_types_cache={ShapedArray(float32[], weak_type=True): RankedTensorType(tensor<f32>), ShapedArray(float32[]): RankedTensorType(tensor<f32>), ShapedArray(int32[]): RankedTensorType(tensor<i32>)}, pallas_lowering_cache={}, traceback_caches=TracebackCaches(traceback_to_location_cache=<jaxlib.mlir._mlir_libs._jax_mlir_ext.TracebackToLocationCache object at 0x7e02eda833f0>, canonical_name_cache={}), lowering_parameters=LoweringParameters(override_lowering_rules=None, global_constant_computation=False, for_export=False, export_ignore_forward_compatibility=False, hoist_constants_as_args=False)), name_stack=NameStack(stack=()), traceback=None, primitive=multiply_add, avals_in=(ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True), ShapedArray(float32[])), avals_out=[ShapedArray(float32[])], tokens_in=<jax._src.interpreters.mlir.TokenSet object at 0x7e02ed915060>, tokens_out=None, const_lowering={}, axis_size_env=None, dim_var_values=[], jaxpr_eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), platforms=None), BlockArgument(<block argument> of type 'tensor<f32>' at index: 0), BlockArgument(<block argument> of type 'tensor<f32>' at index: 1), BlockArgument(<block argument> of type 'tensor<f32>' at index: 2))
    |<- multiply_add_lowering = [<jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x7e02edad3870>]

Notice that first, you evaluate `multiply_add_value_and_jvp` abstractly, which in turn evaluates abstractly both the primal and the tangent evaluation (a total of 3 invocations of the `ma` primitive). Then, you compile the 3 occurrences of the primitive.

### Reverse differentiation[\#](#reverse-differentiation "Link to this heading")

If you attempt now to use reverse differentiation, you’ll notice that JAX starts by using the `multiply_add_value_and_jvp` to compute the forward differentiation for abstract values, but then runs into a `NotImplementedError`.

When computing the reverse differentiation, JAX first performs an abstract evaluation of the forward differentiation code `multiply_add_value_and_jvp` to obtain a trace of primitives that compute the output tangent.

- Observe that JAX performs this abstract evaluation with concrete values for the differentiation point, and abstract values for the tangents.

- Notice that JAX uses the special abstract tangent value `Zero` for the tangent corresponding to the third argument of `ma`. This reflects the fact that you do not differentiate w.r.t. the second argument to `square_add_prim`, which flows to the third argument to `multiply_add_prim`.

- Notice also that during the abstract evaluation of the tangent you pass the value `0.0` as the tangent for the third argument. This is because of the use of the `make_zero` function in the definition of `multiply_add_value_and_jvp`.

    # This is reverse differentiation w.r.t. the first argument of `square_add_prim`
    with expectNotImplementedError():
      api.grad(square_add_prim)(2., 10.)

    call square_add_prim(GradTracer(primal=2.0, typeof(tangent)=f32[]), 10.0)
      call multiply_add_prim(GradTracer(primal=2.0, typeof(tangent)=f32[]), GradTracer(primal=2.0, typeof(tangent)=f32[]), 10.0)
        call multiply_add_value_and_jvp((2.0, 2.0, 10.0), (Traced<~float32[]>, Traced<~float32[]>, Zero(~float32[])))
          Primal evaluation:
          call multiply_add_prim(2.0, 2.0, 10.0)
            call multiply_add_impl(2.0, 2.0, 10.0)
            |<- multiply_add_impl = 14.0
          |<- multiply_add_prim = 14.0
          Tangent evaluation:
          call multiply_add_prim(2.0, Traced<~float32[]>, 0.0)
            call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = Traced<float32[]>
          call multiply_add_prim(Traced<~float32[]>, 2.0, Traced<float32[]>)
            call multiply_add_abstract_eval(~float32[], ~float32[], float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = Traced<float32[]>
        |<- multiply_add_value_and_jvp = (14.0, Traced<float32[]>)
        call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
        |<- multiply_add_abstract_eval = float32[]
        call multiply_add_abstract_eval(~float32[], ~float32[], float32[])
        |<- multiply_add_abstract_eval = float32[]
      |<- multiply_add_prim = GradTracer(primal=14.0, typeof(tangent)=f32[])
    |<- square_add_prim = GradTracer(primal=14.0, typeof(tangent)=f32[])

    Found expected exception:

    Traceback (most recent call last):
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/interpreters/ad.py", line 365, in get_primitive_transpose
        return primitive_transposes[p]
               ~~~~~~~~~~~~~~~~~~~~^^^
    KeyError: multiply_add

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "<frozen runpy>", line 198, in _run_module_as_main
      File "<frozen runpy>", line 88, in _run_code
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/ipykernel_launcher.py", line 18, in <module>
        app.launch_new_instance()
    jax._src.source_info_util.JaxStackTraceBeforeTransformation: NotImplementedError: Transpose rule (for reverse-mode differentiation) for 'multiply_add' not implemented

    The preceding stack trace is the source of the JAX operation that, once transformed by JAX, triggered the following exception.

    --------------------

    The above exception was the direct cause of the following exception:

    Traceback (most recent call last):
      File "/tmp/ipykernel_2055/2155094905.py", line 3, in <module>
        api.grad(square_add_prim)(2., 10.)
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/traceback_util.py", line 195, in reraise_with_filtered_traceback
        return fun(*args, **kwargs)  # pyrefly: ignore[not-callable]
               ^^^^^^^^^^^^^^^^^^^^
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/api.py", line 469, in grad_f
        _, g = value_and_grad_f(*args, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    NotImplementedError: Transpose rule (for reverse-mode differentiation) for 'multiply_add' not implemented

The above error is because there is a missing piece for JAX to be able to use the forward differentiation code to compute reverse differentiation.

#### Transposition[\#](#transposition "Link to this heading")

As previously explained, when computing reverse differentiation, JAX obtains a trace of primitives that compute the tangent using forward differentiation. Then, **JAX interprets this trace abstractly backwards** and for each primitive it applies a **transposition rule**.

To understand what is going on, consider a simpler example of the function `f(x,`` ``y)`` ``=`` ``x`` ``*`` ``y`` ``+`` ``y`. Assume, you need to differentiate at the point `(2.,`` ``4.)`. JAX will produce the following JVP tangent calculation of `ft` from the tangents of the input `xt` and `yt`:

       a = xt * 4.
       b = 2. * yt
       c = a + b
       ft = c + yt

By construction, the tangent calculation is always linear in the input tangents. The only non-linear operator that may arise in the tangent calculation is multiplication, but then one of the operands is constant.

JAX will produce the reverse differentiation computation by processing the JVP computation backwards. For each operation in the tangent computation, it accumulates the cotangents of the variables used by the operation, using the cotangent of the result of the operation:

      # Initialize cotangents of inputs and intermediate variables:
      xct = yct = act = bct = cct = 0.
      # Initialize cotangent of the output:
      fct = 1.
      # Process `ft = c + yt`:
      cct += fct
      yct += fct
      # Process `c = a + b`:
      act += cct
      bct += cct
      # Process `b = 2. * yt`:
      yct += 2. * bct
      # Process `a = xt * 4.`:
      xct += act * 4.

One can verify that this computation produces `xct`` ``=`` ``4.` and `yct`` ``=`` ``3.`, which are the partial derivatives of the function `f`.

JAX knows for each primitive that may appear in a JVP calculation how to transpose it. Conceptually, if the primitive `p(x,`` ``y,`` ``z)` is linear in the arguments `y` and `z` for a constant value of `x`, e.g., `p(x,`` ``y,`` ``z)`` ``=`` ``y*cy`` ``+`` ``z*cz`, then the transposition of the primitive is:

    p_transpose(out_ct, x, _, _) = (None, out_ct*cy, out_ct*cz)

Notice that `p_transpose` takes the cotangent of the output of the primitive and a value corresponding to each argument of the primitive. For the linear arguments, the transposition gets an undefined `_` value, and for the other arguments it gets the actual constants. The transposition returns a cotangent value for each argument of the primitive, with the value `None` returned for the constant arguments.

In particular:

     add_transpose(out_ct, _, _) = (out_ct, out_ct)
     mult_transpose(out_ct, x, _) = (None, x * out_ct)
     mult_transpose(out_ct, _, y) = (out_ct * y, None)

    @trace("multiply_add_transpose")
    def multiply_add_transpose(ct, x, y, z):
      """Evaluates the transpose of a linear primitive.

      This method is only used when computing the backward gradient following
      `value_and_jvp`, and is only needed for primitives that are used in the JVP
      calculation for some other primitive. You need a transposition for `multiply_add_prim`,
      because you have used `multiply_add_prim` in the computation of the `output_tangent` in
      `multiply_add_value_and_jvp`.

      In this case, multiply_add is not a linear primitive. However, it is used linearly
      w.r.t. tangents in `multiply_add_value_and_jvp`:
           `output_tangent(xt, yt, zt) = multiply_add_prim(xt, y, multiply_add_prim(x, yt, zt))`.

      Always one of the first two multiplicative arguments is a constant.

      Args:
          ct: The cotangent of the output of the primitive.
          x, y, z: The values of the arguments. The arguments that are used linearly
            get an ad.UndefinedPrimal value. The other arguments get a constant
            value.

      Returns:
          A tuple with the cotangent of the inputs, with the value None
          corresponding to the constant arguments.
      """
      if not ad.is_undefined_primal(x):
        # This use of multiply_add is with a constant "x".
        assert ad.is_undefined_primal(y)
        ct_y = ad.Zero(y.aval) if type(ct) is ad.Zero else multiply_add_prim(x, ct, lax.full_like(x, 0))
        res = None, ct_y, ct
      else:
        # This use of multiply_add is with a constant "y".
        assert ad.is_undefined_primal(x)
        ct_x = ad.Zero(x.aval) if type(ct) is ad.Zero else multiply_add_prim(ct, y, lax.full_like(y, 0))
        res = ct_x, None, ct
      return res

    ad.primitive_transposes[multiply_add_p] = multiply_add_transpose

Now you can complete the run of the `grad`:

    assert api.grad(square_add_prim)(2., 10.) == 4.

    call square_add_prim(GradTracer(primal=2.0, typeof(tangent)=f32[]), 10.0)
      call multiply_add_prim(GradTracer(primal=2.0, typeof(tangent)=f32[]), GradTracer(primal=2.0, typeof(tangent)=f32[]), 10.0)
        call multiply_add_value_and_jvp((2.0, 2.0, 10.0), (Traced<~float32[]>, Traced<~float32[]>, Zero(~float32[])))
          Primal evaluation:
          call multiply_add_prim(2.0, 2.0, 10.0)
            call multiply_add_impl(2.0, 2.0, 10.0)
            |<- multiply_add_impl = 14.0
          |<- multiply_add_prim = 14.0
          Tangent evaluation:
          call multiply_add_prim(2.0, Traced<~float32[]>, 0.0)
            call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = Traced<float32[]>
          call multiply_add_prim(Traced<~float32[]>, 2.0, Traced<float32[]>)
            call multiply_add_abstract_eval(~float32[], ~float32[], float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = Traced<float32[]>
        |<- multiply_add_value_and_jvp = (14.0, Traced<float32[]>)
      |<- multiply_add_prim = GradTracer(primal=14.0, typeof(tangent)=f32[])
    |<- square_add_prim = GradTracer(primal=14.0, typeof(tangent)=f32[])
    call multiply_add_transpose(1.0, UndefinedPrimal(~float32[]), 2.0, UndefinedPrimal(float32[]))
      call multiply_add_prim(1.0, 2.0, 0.0)
        call multiply_add_impl(1.0, 2.0, 0.0)
        |<- multiply_add_impl = 2.0
      |<- multiply_add_prim = 2.0
    |<- multiply_add_transpose = (2.0, None, 1.0)
    call multiply_add_transpose(1.0, 2.0, UndefinedPrimal(~float32[]), 0.0)
      call multiply_add_prim(2.0, 1.0, 0.0)
        call multiply_add_impl(2.0, 1.0, 0.0)
        |<- multiply_add_impl = 2.0
      |<- multiply_add_prim = 2.0
    |<- multiply_add_transpose = (None, 2.0, 1.0)

Notice the two calls to `multiply_add_transpose`. They correspond to the two uses of `multiply_add_prim` in the computation of the `output_tangent` in `multiply_add_value_and_jvp`. The first call to transpose corresponds to the last use of `multiply_add_prim`: `multiply_add_prim(xt,`` ``y,`` ``...)` where `y` is the constant `2.0`.

#### JIT of reverse differentiation[\#](#jit-of-reverse-differentiation "Link to this heading")

Notice that the abstract evaluation of the `multiply_add_value_and_jvp` is using only abstract values. Meanwhile, in the absence of JIT, you used `ConcreteArray`.

    assert api.jit(api.grad(square_add_prim))(2., 10.) == 4.

    call square_add_prim(GradTracer(primal=JitTracer(~float32[]), typeof(tangent)=f32[]), JitTracer(~float32[]))
      call multiply_add_prim(GradTracer(primal=JitTracer(~float32[]), typeof(tangent)=f32[]), GradTracer(primal=JitTracer(~float32[]), typeof(tangent)=f32[]), JitTracer(~float32[]))
        call multiply_add_value_and_jvp((JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[])), (Traced<~float32[]>, Traced<~float32[]>, Zero(~float32[])))
          Primal evaluation:
          call multiply_add_prim(JitTracer(~float32[]), JitTracer(~float32[]), JitTracer(~float32[]))
            call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = JitTracer(float32[])
          Tangent evaluation:
          call multiply_add_prim(JitTracer(~float32[]), Traced<~float32[]>, JitTracer(~float32[]))
            call multiply_add_abstract_eval(~float32[], ~float32[], ~float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = Traced<float32[]>
          call multiply_add_prim(Traced<~float32[]>, JitTracer(~float32[]), Traced<float32[]>)
            call multiply_add_abstract_eval(~float32[], ~float32[], float32[])
            |<- multiply_add_abstract_eval = float32[]
          |<- multiply_add_prim = Traced<float32[]>
        |<- multiply_add_value_and_jvp = (JitTracer(float32[]), Traced<float32[]>)
      |<- multiply_add_prim = GradTracer(primal=JitTracer(float32[]), typeof(tangent)=f32[])
    |<- square_add_prim = GradTracer(primal=JitTracer(float32[]), typeof(tangent)=f32[])
    call multiply_add_transpose(JitTracer(float32[]), UndefinedPrimal(~float32[]), JitTracer(~float32[]), UndefinedPrimal(float32[]))
      call multiply_add_prim(JitTracer(float32[]), JitTracer(~float32[]), JitTracer(~float32[]))
        call multiply_add_abstract_eval(float32[], ~float32[], ~float32[])
        |<- multiply_add_abstract_eval = float32[]
      |<- multiply_add_prim = JitTracer(float32[])
    |<- multiply_add_transpose = (JitTracer(float32[]), None, JitTracer(float32[]))
    call multiply_add_transpose(JitTracer(float32[]), JitTracer(~float32[]), UndefinedPrimal(~float32[]), JitTracer(~float32[]))
      call multiply_add_prim(JitTracer(~float32[]), JitTracer(float32[]), JitTracer(~float32[]))
        call multiply_add_abstract_eval(~float32[], float32[], ~float32[])
        |<- multiply_add_abstract_eval = float32[]
      |<- multiply_add_prim = JitTracer(float32[])
    |<- multiply_add_transpose = (None, JitTracer(float32[]), JitTracer(float32[]))
    call multiply_add_lowering(LoweringRuleContext(module_context=ModuleContext(context=<jax._src.interpreters.mlir.JaxIrContext object at 0x7e02ed9565d0>, module=<jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7e02ed95cdb0>, ip=<jaxlib.mlir._mlir_libs._mlir.ir.InsertionPoint object at 0x7e02ed95d200>, symbol_table=<jaxlib.mlir._mlir_libs._mlir.ir.SymbolTable object at 0x7e02ed917b40>, platforms=('cpu',), backend=<jaxlib._jax.Client object at 0x7e02f5ebe740>, axis_context=ShardingContext(num_devices=1, device_assignment=None, abstract_mesh=None), keepalives=[], channel_iterator=count(2), host_callbacks=[], shape_poly_state=<jax._src.interpreters.mlir.ShapePolyLoweringState object at 0x7e02ed917d70>, all_default_mem_kind=True, lowering_cache={}, cached_primitive_lowerings={}, sharding_attr_cache={}, aval_to_ir_types_cache={ShapedArray(float32[], weak_type=True): RankedTensorType(tensor<f32>), ShapedArray(float32[]): RankedTensorType(tensor<f32>), ShapedArray(int32[]): RankedTensorType(tensor<i32>)}, pallas_lowering_cache={}, traceback_caches=TracebackCaches(traceback_to_location_cache=<jaxlib.mlir._mlir_libs._jax_mlir_ext.TracebackToLocationCache object at 0x7e02ed930ae0>, canonical_name_cache={}), lowering_parameters=LoweringParameters(override_lowering_rules=None, global_constant_computation=False, for_export=False, export_ignore_forward_compatibility=False, hoist_constants_as_args=False)), name_stack=NameStack(stack=()), traceback=None, primitive=multiply_add, avals_in=(ShapedArray(float32[]), ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True)), avals_out=[ShapedArray(float32[])], tokens_in=<jax._src.interpreters.mlir.TokenSet object at 0x7e02ed9647c0>, tokens_out=None, const_lowering={}, axis_size_env=None, dim_var_values=[], jaxpr_eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), platforms=None), BlockArgument(<block argument> of type 'tensor<f32>' at index: 0), BlockArgument(<block argument> of type 'tensor<f32>' at index: 1), BlockArgument(<block argument> of type 'tensor<f32>' at index: 2))
    |<- multiply_add_lowering = [<jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x7e02edac8330>]
    call multiply_add_lowering(LoweringRuleContext(module_context=ModuleContext(context=<jax._src.interpreters.mlir.JaxIrContext object at 0x7e02ed9565d0>, module=<jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7e02ed95cdb0>, ip=<jaxlib.mlir._mlir_libs._mlir.ir.InsertionPoint object at 0x7e02ed95d200>, symbol_table=<jaxlib.mlir._mlir_libs._mlir.ir.SymbolTable object at 0x7e02ed917b40>, platforms=('cpu',), backend=<jaxlib._jax.Client object at 0x7e02f5ebe740>, axis_context=ShardingContext(num_devices=1, device_assignment=None, abstract_mesh=None), keepalives=[], channel_iterator=count(2), host_callbacks=[], shape_poly_state=<jax._src.interpreters.mlir.ShapePolyLoweringState object at 0x7e02ed917d70>, all_default_mem_kind=True, lowering_cache={LoweringCacheKey(primitive=multiply_add, eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), avals_in=(ShapedArray(float32[]), ShapedArray(float32[], weak_type=True), ShapedArray(float32[], weak_type=True)), effects=frozenset(), params=(), platforms=('cpu',)): LoweringCacheValue(func=<jaxlib.mlir.dialects.func.FuncOp object at 0x7e02ed95d720>, flat_output_types=[RankedTensorType(tensor<f32>)], output_treedef=PyTreeDef([*]), const_args=(), const_arg_avals=(), inline=True)}, cached_primitive_lowerings={}, sharding_attr_cache={}, aval_to_ir_types_cache={ShapedArray(float32[], weak_type=True): RankedTensorType(tensor<f32>), ShapedArray(float32[]): RankedTensorType(tensor<f32>), ShapedArray(int32[]): RankedTensorType(tensor<i32>)}, pallas_lowering_cache={}, traceback_caches=TracebackCaches(traceback_to_location_cache=<jaxlib.mlir._mlir_libs._jax_mlir_ext.TracebackToLocationCache object at 0x7e02ed930ae0>, canonical_name_cache={}), lowering_parameters=LoweringParameters(override_lowering_rules=None, global_constant_computation=False, for_export=False, export_ignore_forward_compatibility=False, hoist_constants_as_args=False)), name_stack=NameStack(stack=()), traceback=None, primitive=multiply_add, avals_in=(ShapedArray(float32[], weak_type=True), ShapedArray(float32[]), ShapedArray(float32[], weak_type=True)), avals_out=[ShapedArray(float32[])], tokens_in=<jax._src.interpreters.mlir.TokenSet object at 0x7e02ed9648b0>, tokens_out=None, const_lowering={}, axis_size_env=None, dim_var_values=[], jaxpr_eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), platforms=None), BlockArgument(<block argument> of type 'tensor<f32>' at index: 0), BlockArgument(<block argument> of type 'tensor<f32>' at index: 1), BlockArgument(<block argument> of type 'tensor<f32>' at index: 2))
    |<- multiply_add_lowering = [<jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x7e02ed9633b0>]

### Batching[\#](#batching "Link to this heading")

The batching transformation takes a point-wise computation and turns it into a computation on vectors. If you try it right now, you will get a `NotImplementedError`:

    # The arguments are two vectors instead of two scalars.
    with expectNotImplementedError():
      api.vmap(square_add_prim, in_axes=0, out_axes=0)(np.array([2., 3.]),
                                                   np.array([10., 20.]))

    call square_add_prim(Traced<float32[]>, Traced<float32[]>)
      call multiply_add_prim(Traced<float32[]>, Traced<float32[]>, Traced<float32[]>)

    Found expected exception:

    Traceback (most recent call last):
      File "/tmp/ipykernel_2055/1080163607.py", line 3, in <module>
        api.vmap(square_add_prim, in_axes=0, out_axes=0)(np.array([2., 3.]),
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/traceback_util.py", line 195, in reraise_with_filtered_traceback
        return fun(*args, **kwargs)  # pyrefly: ignore[not-callable]
               ^^^^^^^^^^^^^^^^^^^^
      File "/home/docs/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/api.py", line 1186, in vmap_f
        out_flat, inferred_out_axes = batching.batch(
                                      ^^^^^^^^^^^^^^^
    NotImplementedError: Batching rule for 'multiply_add' not implemented

You need to instruct JAX how to evaluate the batched version of the primitive. In this particular case, the `multiply_add_prim` already operates pointwise for any dimension of input vectors, so the batched version can use the same `multiply_add_prim` implementation.

    from jax.interpreters import batching

    @trace("multiply_add_batch")
    def multiply_add_batch(vector_arg_values, batch_axes):
      """Computes the batched version of the primitive.

      This must be a JAX-traceable function.

      Since the `multiply_add primitive` already operates point-wise on arbitrary
      dimension tensors, to batch it you can use the primitive itself. This works as
      long as both the inputs have the same dimensions and are batched along the
      same axes. The result is batched along the axis that the inputs are batched.

      Args:
        vector_arg_values: A tuple of two arguments, each being a tensor of matching
          shape.
        batch_axes: The axes that are being batched. See vmap documentation.

      Returns:
        A tuple of the result, and the result axis that was batched.
      """
      assert batch_axes[0] == batch_axes[1]
      assert batch_axes[0] == batch_axes[2]
      _trace("Using multiply_add to compute the batch:")
      res = multiply_add_prim(*vector_arg_values)
      return res, batch_axes[0]


    batching.primitive_batchers[multiply_add_p] = multiply_add_batch

    assert np.allclose(api.vmap(square_add_prim, in_axes=0, out_axes=0)(
      np.array([2., 3.]),
      np.array([10., 20.])),
      [14., 29.])

    call square_add_prim(Traced<float32[]>, Traced<float32[]>)
      call multiply_add_prim(Traced<float32[]>, Traced<float32[]>, Traced<float32[]>)
        call multiply_add_batch(([2. 3.], [2. 3.], [10. 20.]), (0, 0, 0))
          Using multiply_add to compute the batch:
          call multiply_add_prim([2. 3.], [2. 3.], [10. 20.])
            call multiply_add_impl([2. 3.], [2. 3.], [10. 20.])
            |<- multiply_add_impl = [14. 29.]
          |<- multiply_add_prim = [14. 29.]
        |<- multiply_add_batch = ([14. 29.], 0)
      |<- multiply_add_prim = Traced<float32[]>
    |<- square_add_prim = Traced<float32[]>

#### JIT of batching[\#](#jit-of-batching "Link to this heading")

Below is an example of applying JIT to batching:

    assert np.allclose(api.jit(api.vmap(square_add_prim, in_axes=0, out_axes=0))
                        (np.array([2., 3.]),
                         np.array([10., 20.])),
                        [14., 29.])

    call square_add_prim(Traced<float32[]>, Traced<float32[]>)
      call multiply_add_prim(Traced<float32[]>, Traced<float32[]>, Traced<float32[]>)
        call multiply_add_batch((JitTracer(float32[2]), JitTracer(float32[2]), JitTracer(float32[2])), (0, 0, 0))
          Using multiply_add to compute the batch:
          call multiply_add_prim(JitTracer(float32[2]), JitTracer(float32[2]), JitTracer(float32[2]))
            call multiply_add_abstract_eval(float32[2], float32[2], float32[2])
            |<- multiply_add_abstract_eval = float32[2]
          |<- multiply_add_prim = JitTracer(float32[2])
        |<- multiply_add_batch = (JitTracer(float32[2]), 0)
      |<- multiply_add_prim = Traced<float32[]>
    |<- square_add_prim = Traced<float32[]>
    call multiply_add_lowering(LoweringRuleContext(module_context=ModuleContext(context=<jax._src.interpreters.mlir.JaxIrContext object at 0x7e02ed956b50>, module=<jaxlib.mlir._mlir_libs._mlir.ir.Module object at 0x7e02ed95e020>, ip=<jaxlib.mlir._mlir_libs._mlir.ir.InsertionPoint object at 0x7e02ed95e060>, symbol_table=<jaxlib.mlir._mlir_libs._mlir.ir.SymbolTable object at 0x7e02edb0fae0>, platforms=('cpu',), backend=<jaxlib._jax.Client object at 0x7e02f5ebe740>, axis_context=ShardingContext(num_devices=1, device_assignment=None, abstract_mesh=None), keepalives=[], channel_iterator=count(2), host_callbacks=[], shape_poly_state=<jax._src.interpreters.mlir.ShapePolyLoweringState object at 0x7e02edb0c950>, all_default_mem_kind=True, lowering_cache={}, cached_primitive_lowerings={}, sharding_attr_cache={}, aval_to_ir_types_cache={ShapedArray(float32[2]): RankedTensorType(tensor<2xf32>), ShapedArray(int32[]): RankedTensorType(tensor<i32>)}, pallas_lowering_cache={}, traceback_caches=TracebackCaches(traceback_to_location_cache=<jaxlib.mlir._mlir_libs._jax_mlir_ext.TracebackToLocationCache object at 0x7e02ed930ae0>, canonical_name_cache={}), lowering_parameters=LoweringParameters(override_lowering_rules=None, global_constant_computation=False, for_export=False, export_ignore_forward_compatibility=False, hoist_constants_as_args=False)), name_stack=NameStack(stack=()), traceback=None, primitive=multiply_add, avals_in=(ShapedArray(float32[2]), ShapedArray(float32[2]), ShapedArray(float32[2])), avals_out=[ShapedArray(float32[2])], tokens_in=<jax._src.interpreters.mlir.TokenSet object at 0x7e02ed965120>, tokens_out=None, const_lowering={}, axis_size_env=None, dim_var_values=[], jaxpr_eqn_ctx=JaxprEqnContext(compute_type=None, threefry_partitionable=True, cur_abstract_mesh=AbstractMesh((), axis_types=()), remove_size_one_mesh_axis=False, xla_metadata=None), platforms=None), BlockArgument(<block argument> of type 'tensor<2xf32>' at index: 0), BlockArgument(<block argument> of type 'tensor<2xf32>' at index: 1), BlockArgument(<block argument> of type 'tensor<2xf32>' at index: 2))
    |<- multiply_add_lowering = [<jaxlib.mlir._mlir_libs._mlir.ir.OpResult object at 0x7e02ed96c130>]

[](xla_flags.html "previous page")

previous

XLA compiler flags

[](jaxpr.html "next page")

next

JAX internals: The jaxpr language

Contents

- [Introduction to JAX primitives](#introduction-to-jax-primitives)
- [Using existing JAX primitives](#using-existing-jax-primitives)
- [Defining new JAX primitives](#defining-new-jax-primitives)
  - [Primal evaluation rules](#primal-evaluation-rules)
  - [What happens when you use `jit`](#what-happens-when-you-use-jit)
    - [Abstract evaluation rules](#abstract-evaluation-rules)
    - [XLA Compilation rules](#xla-compilation-rules)
  - [Forward differentiation](#forward-differentiation)
    - [JIT of forward differentiation](#jit-of-forward-differentiation)
  - [Reverse differentiation](#reverse-differentiation)
    - [Transposition](#transposition)
    - [JIT of reverse differentiation](#jit-of-reverse-differentiation)
  - [Batching](#batching)
    - [JIT of batching](#jit-of-batching)

By The JAX authors

© Copyright 2024, The JAX Authors.\
