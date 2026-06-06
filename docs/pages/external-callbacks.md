- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- External callbacks

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/external-callbacks.md "Download source file")
-  .pdf

# External callbacks

## Contents

- [Why callbacks?](#why-callbacks)
- [Flavors of callback](#flavors-of-callback)
  - [Exploring `pure_callback`](#exploring-pure-callback)
    - [`pure_callback` and exceptions](#pure-callback-and-exceptions)
  - [Exploring `io_callback`](#exploring-io-callback)
  - [Exploring `debug.callback`](#exploring-debug-callback)
- [Example: `pure_callback` with `custom_jvp`](#example-pure-callback-with-custom-jvp)

# External callbacks[\#](#external-callbacks "Link to this heading")

This tutorial outlines how you can use various callback functions, which allow JAX runtimes to execute Python code on the host. Examples of JAX callbacks are `jax.pure_callback`, `jax.experimental.io_callback` and `jax.debug.callback`. You can use them even while running under JAX transformations, including [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"), [`grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad").

## Why callbacks?[\#](#why-callbacks "Link to this heading")

A callback routine is a way to perform **host-side** execution of code at runtime. As a simple example, suppose you’d like to print the *value* of some variable during the course of a computation. Using a simple Python [`print()`](https://docs.python.org/3/library/functions.html#print "(in Python v3.14)") statement, it looks like this:

    import jax

    @jax.jit
    def f(x):
      y = x + 1
      print("intermediate value: {}".format(y))
      return y * 2

    result = f(2)

    intermediate value: JitTracer(~int32[])

What is printed is not the runtime value, but the trace-time abstract value (if you’re not familiar with *tracing* in JAX, a good primer can be found in [Tracing](key-concepts.html#key-concepts-tracing).

To print the value at runtime, you need a callback, for example [`jax.debug.print()`](_autosummary/jax.debug.print.html#jax.debug.print "jax.debug.print") (you can learn more about debugging in [Introduction to debugging](debugging.html#debugging)):

    @jax.jit
    def f(x):
      y = x + 1
      jax.debug.print("intermediate value: {}", y)
      return y * 2

    result = f(2)

    intermediate value: 3

This works by passing the runtime value of `y` as a CPU [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") back to the host process, where the host can print it.

## Flavors of callback[\#](#flavors-of-callback "Link to this heading")

In earlier versions of JAX, there was only one kind of callback available, implemented in `jax.experimental.host_callback()`. The `host_callback` routines had some deficiencies, and are now deprecated in favor of several callbacks designed for different situations:

- [`jax.pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback"): appropriate for pure functions: i.e. functions with no side effects. See [Exploring pure_callback](#external-callbacks-exploring-pure-callback).

- [`jax.experimental.io_callback()`](_autosummary/jax.experimental.io_callback.html#jax.experimental.io_callback "jax.experimental.io_callback"): appropriate for impure functions: e.g. functions which read or write data to disk. See [Exploring io_callback](#external-callbacks-exploring-io-callback).

- [`jax.debug.callback()`](_autosummary/jax.debug.callback.html#jax.debug.callback "jax.debug.callback"): appropriate for functions that should reflect the execution behavior of the compiler. See [Exploring debug.callback](#external-callbacks-exploring-debug-callback).

(The [`jax.debug.print()`](_autosummary/jax.debug.print.html#jax.debug.print "jax.debug.print") function you used previously is a wrapper around [`jax.debug.callback()`](_autosummary/jax.debug.callback.html#jax.debug.callback "jax.debug.callback")).

From the user perspective, these three flavors of callback are mainly distinguished by what transformations and compiler optimizations they allow.

| callback function | supports return value | `jit` | `vmap` | `grad` | `scan`/`while_loop` | guaranteed execution |
|----|----|----|----|----|----|----|
| [`jax.pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback") | ✅ | ✅ | ✅ | ❌¹ | ✅ | ❌ |
| [`jax.experimental.io_callback()`](_autosummary/jax.experimental.io_callback.html#jax.experimental.io_callback "jax.experimental.io_callback") | ✅ | ✅ | ✅/❌² | ❌ | ✅³ | ✅ |
| [`jax.debug.callback()`](_autosummary/jax.debug.callback.html#jax.debug.callback "jax.debug.callback") | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ |

¹ `jax.pure_callback` can be used with `custom_jvp` to make it compatible with autodiff

² `jax.experimental.io_callback` is compatible with `vmap` only if `ordered=False`.

³ Note that `vmap` of `scan`/`while_loop` of `io_callback` has complicated semantics, and its behavior may change in future releases.

### Exploring `pure_callback`[\#](#exploring-pure-callback "Link to this heading")

[`jax.pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback") is generally the callback function you should reach for when you want host-side execution of a pure function: i.e. a function that has no side-effects (such as printing values, reading data from disk, updating a global state, etc.).

The function you pass to [`jax.pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback") need not actually be pure, but it will be assumed pure by JAX’s transformations and higher-order functions, which means that it may be silently elided or called multiple times.

    import jax
    import jax.numpy as jnp
    import numpy as np

    def f_host(x):
      # call a numpy (not jax.numpy) operation:
      return np.sin(x).astype(x.dtype)

    def f(x):
      result_shape = jax.ShapeDtypeStruct.like(x)
      return jax.pure_callback(f_host, result_shape, x, vmap_method='sequential')

    x = jnp.arange(5.0)
    f(x)

    Array([ 0.       ,  0.841471 ,  0.9092974,  0.14112  , -0.7568025],      dtype=float32)

Because `pure_callback` can be elided or duplicated, it is compatible out-of-the-box with transformations like `jit` as well as higher-order primitives like `scan` and `while_loop`:”

    jax.jit(f)(x)

    Array([ 0.       ,  0.841471 ,  0.9092974,  0.14112  , -0.7568025],      dtype=float32)

    def body_fun(_, x):
      return _, f(x)
    jax.lax.scan(body_fun, None, jnp.arange(5.0))[1]

    Array([ 0.       ,  0.841471 ,  0.9092974,  0.14112  , -0.7568025],      dtype=float32)

Because we specified a `vmap_method` in the `pure_callback` function call, it will also be compatible with `vmap`:

    jax.vmap(f)(x)

    Array([ 0.       ,  0.841471 ,  0.9092974,  0.14112  , -0.7568025],      dtype=float32)

However, because there is no way for JAX to introspect the content of the callback, `pure_callback` has undefined autodiff semantics:

    jax.grad(f)(x)

    ValueError: Pure callbacks do not support JVP. Please use `jax.custom_jvp` to use callbacks while taking gradients.

For an example of using `pure_callback` with [`jax.custom_jvp()`](_autosummary/jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp"), see *Example: `pure_callback` with `custom_jvp`* below.

By design functions passed to `pure_callback` are treated as if they have no side-effects: one consequence of this is that if the output of the function is not used, the compiler may eliminate the callback entirely:

    def print_something():
      print('printing something')
      return np.int32(0)

    @jax.jit
    def f1():
      return jax.pure_callback(print_something, np.int32(0))
    f1();

    printing something

    @jax.jit
    def f2():
      jax.pure_callback(print_something, np.int32(0))
      return 1.0
    f2();

In `f1`, the output of the callback is used in the return value of the function, so the callback is executed and we see the printed output. In `f2` on the other hand, the output of the callback is unused, and so the compiler notices this and eliminates the function call. These are the correct semantics for a callback to a function with no side-effects.

#### `pure_callback` and exceptions[\#](#pure-callback-and-exceptions "Link to this heading")

In the context of JAX transformations, Python runtime exceptions should be considered side-effects: this means that intentionally raising an error within a `pure_callback` breaks the API contract, and the behavior of the resulting program is undefined. In particular, the manner in which such a program halts will generally depend on the backend, and the details of that behavior may change in future releases.

Additionally, passing impure functions to `pure_callback` may result in unexpected behavior during transformations like [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") or [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"), because the transformation rules for `pure_callback` are defined under the assumption that the callback function is pure. Here’s one simple example of an impure callback behaving unexpectedly under `vmap`:

    import jax
    import jax.numpy as jnp

    def raise_via_callback(x):
      def _raise(x):
        raise ValueError(f"value of x is {x}")
      return jax.pure_callback(_raise, x, x)

    def raise_if_negative(x):
      return jax.lax.cond(x < 0, raise_via_callback, lambda x: x, x)

    x_batch = jnp.arange(4)

    [raise_if_negative(x) for x in x_batch]  # does not raise

    jax.vmap(raise_if_negative)(x_batch)  # ValueError: value of x is 0

To avoid this and similar unexpected behavior, we recommend not attempting to use `pure_callback` to raise runtime errors.

### Exploring `io_callback`[\#](#exploring-io-callback "Link to this heading")

In contrast to [`jax.pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback"), [`jax.experimental.io_callback()`](_autosummary/jax.experimental.io_callback.html#jax.experimental.io_callback "jax.experimental.io_callback") is explicitly meant to be used with impure functions, i.e. functions that do have side-effects.

As an example, here is a callback to a global host-side numpy random generator. This is an impure operation because a side-effect of generating a random number in numpy is that the random state is updated (Please note that this is meant as a toy example of `io_callback` and not necessarily a recommended way of generating random numbers in JAX!).

    from jax.experimental import io_callback
    from functools import partial

    global_rng = np.random.default_rng(0)

    def host_side_random_like(x):
      """Generate a random array like x using the global_rng state"""
      # We have two side-effects here:
      # - printing the shape and dtype
      # - calling global_rng, thus updating its state
      print(f'generating {x.dtype}{list(x.shape)}')
      return global_rng.uniform(size=x.shape).astype(x.dtype)

    @jax.jit
    def numpy_random_like(x):
      return io_callback(host_side_random_like, x, x)

    x = jnp.zeros(5)
    numpy_random_like(x)

    generating float32[5]

    Array([0.6369617 , 0.26978672, 0.04097353, 0.01652764, 0.8132702 ],      dtype=float32)

The `io_callback` is compatible with `vmap` by default:

    jax.vmap(numpy_random_like)(x)

    generating float32[]
    generating float32[]
    generating float32[]
    generating float32[]
    generating float32[]

    Array([0.91275555, 0.60663575, 0.72949654, 0.543625  , 0.9350724 ],      dtype=float32)

Note, however, that this may execute the mapped callbacks in any order. So, for example, if you ran this on a GPU, the order of the mapped outputs might differ from run to run.

If it is important that the order of callbacks be preserved, you can set `ordered=True`, in which case attempting to `vmap` will raise an error:

    @jax.jit
    def numpy_random_like_ordered(x):
      return io_callback(host_side_random_like, x, x, ordered=True)

    jax.vmap(numpy_random_like_ordered)(x)

    ValueError: Cannot `vmap` ordered IO callback.

On the other hand, `scan` and `while_loop` work with `io_callback` regardless of whether ordering is enforced:

    def body_fun(_, x):
      return _, numpy_random_like_ordered(x)
    jax.lax.scan(body_fun, None, jnp.arange(5.0))[1]

    generating float32[]
    generating float32[]
    generating float32[]
    generating float32[]
    generating float32[]

    Array([0.81585354, 0.0027385 , 0.8574043 , 0.03358557, 0.72965544],      dtype=float32)

Like `pure_callback`, `io_callback` fails under automatic differentiation if it is passed a differentiated variable:

    jax.grad(numpy_random_like)(x)

    ValueError: IO callbacks do not support JVP.

However, if the callback is not dependent on a differentiated variable, it will execute:

    @jax.jit
    def f(x):
      io_callback(lambda: print('hello'), None)
      return x

    jax.grad(f)(1.0);

    hello

Unlike `pure_callback`, the compiler will not remove the callback execution in this case, even though the output of the callback is unused in the subsequent computation.

### Exploring `debug.callback`[\#](#exploring-debug-callback "Link to this heading")

Both `pure_callback` and `io_callback` enforce some assumptions about the purity of the function they’re calling, and limit in various ways what JAX transforms and compilation machinery may do. `debug.callback` essentially assumes *nothing* about the callback function, such that the action of the callback reflects exactly what JAX is doing during the course of a program. Further, `debug.callback` *cannot* return any value to the program.

    from jax import debug

    def log_value(x):
      # This could be an actual logging call; we'll use
      # print() for demonstration
      print("log:", x)

    @jax.jit
    def f(x):
      debug.callback(log_value, x)
      return x

    f(1.0);

    log: 1.0

The debug callback is compatible with `vmap`:

    x = jnp.arange(5.0)
    jax.vmap(f)(x);

    log: 0.0
    log: 1.0
    log: 2.0
    log: 3.0
    log: 4.0

And is also compatible with `grad` and other autodiff transformations

    jax.grad(f)(1.0);

    log: 1.0

This can make `debug.callback` more useful for general-purpose debugging than either `pure_callback` or `io_callback`.

## Example: `pure_callback` with `custom_jvp`[\#](#example-pure-callback-with-custom-jvp "Link to this heading")

One powerful way to take advantage of [`jax.pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback") is to combine it with [`jax.custom_jvp`](_autosummary/jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp"). (Refer to [Custom derivative rules for JAX-transformable Python functions](notebooks/Custom_derivative_rules_for_Python_code.html#advanced-autodiff-custom-derivative-rules) for more details on [`jax.custom_jvp()`](_autosummary/jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp")).

Suppose you want to create a JAX-compatible wrapper for a scipy or numpy function that is not yet available in the `jax.scipy` or [`jax.numpy`](jax.numpy.html#module-jax.numpy "jax.numpy") wrappers.

Here, we’ll consider creating a wrapper for the Bessel function of the first kind, available in `scipy.special.jv`. You can start by defining a straightforward [`pure_callback()`](_autosummary/jax.pure_callback.html#jax.pure_callback "jax.pure_callback"):

    import jax
    import jax.numpy as jnp
    import scipy.special

    def jv(v, z):
      v, z = jnp.asarray(v), jnp.asarray(z)

      # Require the order v to be integer type: this simplifies
      # the JVP rule below.
      assert jnp.issubdtype(v.dtype, jnp.integer)

      # Promote the input to inexact (float/complex).
      # Note that jnp.result_type() accounts for the enable_x64 flag.
      z = z.astype(jnp.result_type(float, z.dtype))

      # Wrap scipy function to return the expected dtype.
      _scipy_jv = lambda v, z: scipy.special.jv(v, z).astype(z.dtype)

      # Define the expected shape & dtype of output.
      result_shape_dtype = jax.ShapeDtypeStruct(
          shape=jnp.broadcast_shapes(v.shape, z.shape),
          dtype=z.dtype)

      # Use vmap_method="broadcast_all" because scipy.special.jv handles broadcasted inputs.
      return jax.pure_callback(_scipy_jv, result_shape_dtype, v, z, vmap_method="broadcast_all")

This lets us call into `scipy.special.jv()` from transformed JAX code, including when transformed by [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit") and [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"):

    from functools import partial
    j1 = partial(jv, 1)
    z = jnp.arange(5.0)

    print(j1(z))

    [ 0.          0.44005057  0.5767248   0.33905897 -0.06604332]

Here is the same result with [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"):

    print(jax.jit(j1)(z))

    [ 0.          0.44005057  0.5767248   0.33905897 -0.06604332]

And here is the same result again with [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap"):

    print(jax.vmap(j1)(z))

    [ 0.          0.44005057  0.5767248   0.33905897 -0.06604332]

However, if you call [`grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"), you will get an error because there is no autodiff rule defined for this function:

    jax.grad(j1)(z)

    ValueError: Pure callbacks do not support JVP. Please use `jax.custom_jvp` to use callbacks while taking gradients.

Let’s define a custom gradient rule for this. Looking at the definition of the [Bessel Function of the First Kind](https://en.wikipedia.org/?title=Bessel_function_of_the_first_kind), you find that there is a relatively straightforward recurrence relationship for the derivative with respect to the argument `z`:

\\\begin{split} d J\_\nu(z) = \left\\ \begin{eqnarray} -J_1(z),\\ &\nu=0\\ \[J\_{\nu - 1}(z) - J\_{\nu + 1}(z)\]/2,\\ &\nu\ne 0 \end{eqnarray}\right. \end{split}\\

The gradient with respect to \\\nu\\ is more complicated, but since we’ve restricted the `v` argument to integer types you don’t need to worry about its gradient for the sake of this example.

You can use [`jax.custom_jvp()`](_autosummary/jax.custom_jvp.html#jax.custom_jvp "jax.custom_jvp") to define this automatic differentiation rule for your callback function:

    jv = jax.custom_jvp(jv)

    @jv.defjvp
    def _jv_jvp(primals, tangents):
      v, z = primals
      _, z_dot = tangents  # Note: v_dot is always 0 because v is integer.
      jv_minus_1, jv_plus_1 = jv(v - 1, z), jv(v + 1, z)
      djv_dz = jnp.where(v == 0, -jv_plus_1, 0.5 * (jv_minus_1 - jv_plus_1))
      return jv(v, z), z_dot * djv_dz

Now computing the gradient of your function will work correctly:

    j1 = partial(jv, 1)
    print(jax.grad(j1)(2.0))

    -0.06447162

Further, since we’ve defined your gradient in terms of `jv` itself, JAX’s architecture means that you get second-order and higher derivatives for free:

    jax.hessian(j1)(2.0)

    Array(-0.4003078, dtype=float32, weak_type=True)

Keep in mind that although this all works correctly with JAX, each call to your callback-based `jv` function will result in passing the input data from the device to the host, and passing the output of `scipy.special.jv()` from the host back to the device.

When running on accelerators like GPU or TPU, this data movement and host synchronization can lead to significant overhead each time `jv` is called.

However, if you are running JAX on a single CPU (where the “host” and “device” are on the same hardware), JAX will generally do this data transfer in a fast, zero-copy fashion, making this pattern a relatively straightforward way to extend JAX’s capabilities.

[](array_refs.html "previous page")

previous

`Ref`: mutable arrays for data plumbing and memory control

[](ffi.html "next page")

next

Foreign function interface (FFI)

Contents

- [Why callbacks?](#why-callbacks)
- [Flavors of callback](#flavors-of-callback)
  - [Exploring `pure_callback`](#exploring-pure-callback)
    - [`pure_callback` and exceptions](#pure-callback-and-exceptions)
  - [Exploring `io_callback`](#exploring-io-callback)
  - [Exploring `debug.callback`](#exploring-debug-callback)
- [Example: `pure_callback` with `custom_jvp`](#example-pure-callback-with-custom-jvp)

By The JAX authors

© Copyright 2024, The JAX Authors.\
