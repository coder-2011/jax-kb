- [](../index.html)
- [Extension guides](../extensions.html)
- Writing custom Jaxpr interpreters in JAX

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .ipynb](../_sources/notebooks/Writing_custom_interpreters_in_Jax.ipynb "Download source file")
-  .pdf

# Writing custom Jaxpr interpreters in JAX

## Contents

- [What is JAX doing?](#what-is-jax-doing)
- [Jaxpr tracer](#jaxpr-tracer)
  - [Why are Jaxprs useful?](#why-are-jaxprs-useful)
- [Your first interpreter: `invert`](#your-first-interpreter-invert)
  - [1. Tracing a function](#tracing-a-function)
  - [2. Evaluating a Jaxpr](#evaluating-a-jaxpr)
  - [Custom `inverse` Jaxpr interpreter](#custom-inverse-jaxpr-interpreter)
- [Exercises for the reader](#exercises-for-the-reader)

# Writing custom Jaxpr interpreters in JAX[\#](#writing-custom-jaxpr-interpreters-in-jax "Link to this heading")

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/Writing_custom_interpreters_in_Jax.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax/blob/main/docs/notebooks/Writing_custom_interpreters_in_Jax.ipynb)

JAX offers several composable function transformations (`jit`, `grad`, `vmap`, etc.) that enable writing concise, accelerated code.

Here we show how to add your own function transformations to the system, by writing a custom Jaxpr interpreter. And we’ll get composability with all the other transformations for free.

**This example uses internal JAX APIs, which may break at any time. Anything not in [the API Documentation](https://docs.jax.dev/en/latest/jax.html) should be assumed internal.**

    import jax
    import jax.numpy as jnp
    from jax import jit, grad, vmap
    from jax import random

## What is JAX doing?[\#](#what-is-jax-doing "Link to this heading")

JAX provides a NumPy-like API for numerical computing which can be used as is, but JAX’s true power comes from composable function transformations. Take the `jit` function transformation, which takes in a function and returns a semantically identical function but is lazily compiled by XLA for accelerators.

    x = random.normal(random.key(0), (5000, 5000))
    def f(w, b, x):
      return jnp.tanh(jnp.dot(x, w) + b)
    fast_f = jit(f)

When we call `fast_f`, what happens? JAX traces the function and constructs an XLA computation graph. The graph is then JIT-compiled and executed. Other transformations work similarly in that they first trace the function and handle the output trace in some way. To learn more about Jax’s tracing machinery, you can refer to the [“How it works”](https://github.com/jax-ml/jax#how-it-works) section in the README.

## Jaxpr tracer[\#](#jaxpr-tracer "Link to this heading")

A tracer of special importance in Jax is the Jaxpr tracer, which records ops into a Jaxpr (Jax expression). A Jaxpr is a data structure that can be evaluated like a mini functional programming language and thus Jaxprs are a useful intermediate representation for function transformation.

To get a first look at Jaxprs, consider the `make_jaxpr` transformation. `make_jaxpr` is essentially a “pretty-printing” transformation: it transforms a function into one that, given example arguments, produces a Jaxpr representation of its computation. `make_jaxpr` is useful for debugging and introspection. Let’s use it to look at how some example Jaxprs are structured.

    def examine_jaxpr(closed_jaxpr):
      jaxpr = closed_jaxpr.jaxpr
      print("invars:", jaxpr.invars)
      print("outvars:", jaxpr.outvars)
      print("constvars:", jaxpr.constvars)
      for eqn in jaxpr.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
      print()
      print("jaxpr:", jaxpr)

    def foo(x):
      return x + 1
    print("foo")
    print("=====")
    examine_jaxpr(jax.make_jaxpr(foo)(5))

    print()

    def bar(w, b, x):
      return jnp.dot(w, x) + b + jnp.ones(5), x
    print("bar")
    print("=====")
    examine_jaxpr(jax.make_jaxpr(bar)(jnp.ones((5, 10)), jnp.ones(5), jnp.ones(10)))

    foo
    =====
    invars: [Var(id=125306726499776):int32[]]
    outvars: [Var(id=125306726500160):int32[]]
    constvars: []
    equation: [Var(id=125306726499776):int32[], Literal(TypedInt(1, dtype=int32))] add [Var(id=125306726500160):int32[]] {}

    jaxpr: { lambda ; a:i32[]. let b:i32[] = add a 1:i32[] in (b,) }

    bar
    =====
    invars: [Var(id=125306726692608):float32[5,10], Var(id=125306726692480):float32[5], Var(id=125306726692352):float32[10]]
    outvars: [Var(id=125306682757760):float32[5], Var(id=125306726692352):float32[10]]
    constvars: []
    equation: [Var(id=125306726692608):float32[5,10], Var(id=125306726692352):float32[10]] dot_general [Var(id=125306682754624):float32[5]] {'dimension_numbers': (((1,), (0,)), ((), ())), 'precision': None, 'preferred_element_type': dtype('float32'), 'out_sharding': None}
    equation: [Var(id=125306682754624):float32[5], Var(id=125306726692480):float32[5]] add [Var(id=125306682757376):float32[5]] {}
    equation: [Literal(1.0)] broadcast_in_dim [Var(id=125306682757312):float32[5]] {'shape': (5,), 'broadcast_dimensions': (), 'sharding': None}
    equation: [Var(id=125306682757376):float32[5], Var(id=125306682757312):float32[5]] add [Var(id=125306682757760):float32[5]] {}

    jaxpr: { lambda ; a:f32[5,10] b:f32[5] c:f32[10]. let
        d:f32[5] = dot_general[
          dimension_numbers=(([1], [0]), ([], []))
          preferred_element_type=float32
        ] a c
        e:f32[5] = add d b
        f:f32[5] = broadcast_in_dim 1.0:f32[]
        g:f32[5] = add e f
      in (g, c) }

- `jaxpr.invars` - the `invars` of a Jaxpr are a list of the input variables to Jaxpr, analogous to arguments in Python functions.

- `jaxpr.outvars` - the `outvars` of a Jaxpr are the variables that are returned by the Jaxpr. Every Jaxpr has multiple outputs.

- `jaxpr.constvars` - the `constvars` are a list of variables that are also inputs to the Jaxpr, but correspond to constants from the trace (we’ll go over these in more detail later).

- `jaxpr.eqns` - a list of equations, which are essentially let-bindings. Each equation is a list of input variables, a list of output variables, and a *primitive*, which is used to evaluate inputs to produce outputs. Each equation also has a `params`, a dictionary of parameters.

Altogether, a Jaxpr encapsulates a simple program that can be evaluated with inputs to produce an output. We’ll go over how exactly to do this later. The important thing to note now is that a Jaxpr is a data structure that can be manipulated and evaluated in whatever way we want.

### Why are Jaxprs useful?[\#](#why-are-jaxprs-useful "Link to this heading")

Jaxprs are simple program representations that are easy to transform. And because Jax lets us stage out Jaxprs from Python functions, it gives us a way to transform numerical programs written in Python.

## Your first interpreter: `invert`[\#](#your-first-interpreter-invert "Link to this heading")

Let’s try to implement a simple function “inverter”, which takes in the output of the original function and returns the inputs that produced those outputs. For now, let’s focus on simple, unary functions which are composed of other invertible unary functions.

Goal:

    def f(x):
      return jnp.exp(jnp.tanh(x))
    f_inv = inverse(f)
    assert jnp.allclose(f_inv(f(1.0)), 1.0)

The way we’ll implement this is by (1) tracing `f` into a Jaxpr, then (2) interpreting the Jaxpr *backwards*. While interpreting the Jaxpr backwards, for each equation we’ll look up the primitive’s inverse in a table and apply it.

### 1. Tracing a function[\#](#tracing-a-function "Link to this heading")

Let’s use `make_jaxpr` to trace a function into a Jaxpr.

    # Importing Jax functions useful for tracing/interpreting.
    from functools import wraps

    from jax import lax
    from jax.extend import core
    from jax._src.util import safe_map

`jax.make_jaxpr` returns a *closed* Jaxpr, which is a Jaxpr that has been bundled with the constants (`literals`) from the trace.

    def f(x):
      return jnp.exp(jnp.tanh(x))

    closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
    print(closed_jaxpr.jaxpr)
    print(closed_jaxpr.literals)

    { lambda ; a:f32[5]. let b:f32[5] = tanh a; c:f32[5] = exp b in (c,) }
    []

### 2. Evaluating a Jaxpr[\#](#evaluating-a-jaxpr "Link to this heading")

Before we write a custom Jaxpr interpreter, let’s first implement the “default” interpreter, `eval_jaxpr`, which evaluates the Jaxpr as-is, computing the same values that the original, un-transformed Python function would.

To do this, we first create an environment to store the values for each of the variables, and update the environment with each equation we evaluate in the Jaxpr.

    def eval_jaxpr(jaxpr, consts, *args):
      # Mapping from variable -> value
      env = {}

      def read(var):
        # Literals are values baked into the Jaxpr
        if type(var) is core.Literal:
          return var.val
        return env[var]

      def write(var, val):
        env[var] = val

      # Bind args and consts to environment
      safe_map(write, jaxpr.invars, args)
      safe_map(write, jaxpr.constvars, consts)

      # Loop through equations and evaluate primitives using `bind`
      for eqn in jaxpr.eqns:
        # Read inputs to equation from environment
        invals = safe_map(read, eqn.invars)
        # `bind` is how a primitive is called
        outvals = eqn.primitive.bind(*invals, **eqn.params)
        # Primitives may return multiple outputs or not
        if not eqn.primitive.multiple_results:
          outvals = [outvals]
        # Write the results of the primitive into the environment
        safe_map(write, eqn.outvars, outvals)
      # Read the final result of the Jaxpr from the environment
      return safe_map(read, jaxpr.outvars)

    closed_jaxpr = jax.make_jaxpr(f)(jnp.ones(5))
    eval_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, jnp.ones(5))

    [Array([2.1416876, 2.1416876, 2.1416876, 2.1416876, 2.1416876], dtype=float32)]

Notice that `eval_jaxpr` will always return a flat list even if the original function does not.

Furthermore, this interpreter does not handle higher-order primitives (like `jit` and `pmap`), which we will not cover in this guide. You can refer to `core.eval_jaxpr` ([link](https://github.com/jax-ml/jax/blob/main/jax/core.py)) to see the edge cases that this interpreter does not cover.

### Custom `inverse` Jaxpr interpreter[\#](#custom-inverse-jaxpr-interpreter "Link to this heading")

An `inverse` interpreter doesn’t look too different from `eval_jaxpr`. We’ll first set up the registry which will map primitives to their inverses. We’ll then write a custom interpreter that looks up primitives in the registry.

It turns out that this interpreter will also look similar to the “transpose” interpreter used in reverse-mode autodifferentiation [found here](https://github.com/jax-ml/jax/blob/main/jax/interpreters/ad.py#L164-L234).

    inverse_registry = {}

We’ll now register inverses for some of the primitives. By convention, primitives in Jax end in `_p` and a lot of the popular ones live in `lax`.

    inverse_registry[lax.exp_p] = jnp.log
    inverse_registry[lax.tanh_p] = jnp.arctanh

`inverse` will first trace the function, then custom-interpret the Jaxpr. Let’s set up a simple skeleton.

    def inverse(fun):
      @wraps(fun)
      def wrapped(*args, **kwargs):
        # Since we assume unary functions, we won't worry about flattening and
        # unflattening arguments.
        closed_jaxpr = jax.make_jaxpr(fun)(*args, **kwargs)
        out = inverse_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out[0]
      return wrapped

Now we just need to define `inverse_jaxpr`, which will walk through the Jaxpr backward and invert primitives when it can.

    def inverse_jaxpr(jaxpr, consts, *args):
      env = {}

      def read(var):
        if type(var) is core.Literal:
          return var.val
        return env[var]

      def write(var, val):
        env[var] = val
      # Args now correspond to Jaxpr outvars
      safe_map(write, jaxpr.outvars, args)
      safe_map(write, jaxpr.constvars, consts)

      # Looping backward
      for eqn in jaxpr.eqns[::-1]:
        #  outvars are now invars
        invals = safe_map(read, eqn.outvars)
        if eqn.primitive not in inverse_registry:
          raise NotImplementedError(
              f"{eqn.primitive} does not have registered inverse.")
        # Assuming a unary function
        outval = inverse_registry[eqn.primitive](*invals)
        safe_map(write, eqn.invars, [outval])
      return safe_map(read, jaxpr.invars)

That’s it!

    def f(x):
      return jnp.exp(jnp.tanh(x))

    f_inv = inverse(f)
    assert jnp.allclose(f_inv(f(1.0)), 1.0)

Importantly, you can trace through a Jaxpr interpreter.

    jax.make_jaxpr(inverse(f))(f(1.))

    { lambda ; a:f32[]. let b:f32[] = log a; c:f32[] = atanh b in (c,) }

That’s all it takes to add a new transformation to a system, and you get composition with all the others for free! For example, we can use `jit`, `vmap`, and `grad` with `inverse`!

    jit(vmap(grad(inverse(f))))((jnp.arange(5) + 1.) / 5.)

    Array([-3.1440797, 15.584931 ,  2.2551253,  1.3155028,  1.       ],      dtype=float32, weak_type=True)

## Exercises for the reader[\#](#exercises-for-the-reader "Link to this heading")

- Handle primitives with multiple arguments where inputs are partially known, for example `lax.add_p`, `lax.mul_p`.

- Handle `xla_call` and `xla_pmap` primitives, which will not work with both `eval_jaxpr` and `inverse_jaxpr` as written.

[](../extensions.html "previous page")

previous

Extension guides

[](../building_on_jax.html "next page")

next

Building on JAX

Contents

- [What is JAX doing?](#what-is-jax-doing)
- [Jaxpr tracer](#jaxpr-tracer)
  - [Why are Jaxprs useful?](#why-are-jaxprs-useful)
- [Your first interpreter: `invert`](#your-first-interpreter-invert)
  - [1. Tracing a function](#tracing-a-function)
  - [2. Evaluating a Jaxpr](#evaluating-a-jaxpr)
  - [Custom `inverse` Jaxpr interpreter](#custom-inverse-jaxpr-interpreter)
- [Exercises for the reader](#exercises-for-the-reader)

By The JAX authors

© Copyright 2024, The JAX Authors.\
