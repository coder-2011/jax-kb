- [](../index.html)
- 🔪 JAX - The Sharp Bits 🔪

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .ipynb](../_sources/notebooks/Common_Gotchas_in_JAX.ipynb "Download source file")
-  .pdf

# 🔪 JAX - The Sharp Bits 🔪

## Contents

- [🔪 Pure functions](#pure-functions)
- [🔪 In-place updates](#in-place-updates)
  - [Array updates: `x.at[idx].set(y)`](#array-updates-x-at-idx-set-y)
  - [Array updates with other operations](#array-updates-with-other-operations)
- [🔪 Using `jax.jit` with class methods](#using-jax-jit-with-class-methods)
  - [Strategy 1: JIT-compiled helper function](#strategy-1-jit-compiled-helper-function)
  - [Strategy 2: Marking `self` as static](#strategy-2-marking-self-as-static)
  - [Strategy 3: Making `CustomClass` a PyTree](#strategy-3-making-customclass-a-pytree)
- [🔪 Out-of-bounds indexing](#out-of-bounds-indexing)
- [🔪 Non-array inputs: NumPy vs. JAX](#non-array-inputs-numpy-vs-jax)
- [🔪 Random numbers](#random-numbers)
- [🔪 Control flow](#control-flow)
- [🔪 Dynamic shapes](#dynamic-shapes)
- [🔪 Debugging NaNs and Infs](#debugging-nans-and-infs)
- [🔪 Double (64bit) precision](#double-64bit-precision)
  - [Caveats](#caveats)
- [🔪 Miscellaneous divergences from NumPy](#miscellaneous-divergences-from-numpy)
- [🔪 Sharp bits covered in tutorials](#sharp-bits-covered-in-tutorials)
- [Fin.](#fin)

# 🔪 JAX - The Sharp Bits 🔪[\#](#jax-the-sharp-bits "Link to this heading")

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jax-ml/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb) [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/jax-ml/jax/blob/main/docs/notebooks/Common_Gotchas_in_JAX.ipynb)

When walking about the countryside of Italy, the people will not hesitate to tell you that **JAX** has [*“una anima di pura programmazione funzionale”*](https://www.sscardapane.it/iaml-backup/jax-intro/).

**JAX** is a language for **expressing** and **composing** **transformations** of numerical programs. **JAX** is also able to **compile** numerical programs for CPU or accelerators (GPU/TPU). JAX works great for many numerical and scientific programs, but **only if they are written with certain constraints** that we describe below.

    import numpy as np
    from jax import jit
    from jax import lax
    from jax import random
    import jax
    import jax.numpy as jnp

## 🔪 Pure functions[\#](#pure-functions "Link to this heading")

JAX transformation and compilation are designed to work only on Python functions that are functionally pure: all the input data is passed through the function parameters, all the results are output through the function results. A pure function will always return the same result if invoked with the same inputs.

Here are some examples of functions that are not functionally pure for which JAX behaves differently than the Python interpreter. Note that these behaviors are not guaranteed by the JAX system; the proper way to use JAX is to use it only on functionally pure Python functions.

    def impure_print_side_effect(x):
      print("Executing function")  # This is a side-effect
      return x

    # The side-effects appear during the first run
    print ("First call: ", jit(impure_print_side_effect)(4.))

    # Subsequent runs with parameters of same type and shape may not show the side-effect
    # This is because JAX now invokes a cached compilation of the function
    print ("Second call: ", jit(impure_print_side_effect)(5.))

    # JAX re-runs the Python function when the type or shape of the argument changes
    print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))

    Executing function
    First call:  4.0
    Second call:  5.0
    Executing function
    Third call, different type:  [5.]

    g = 0.
    def impure_uses_globals(x):
      return x + g

    # JAX captures the value of the global during the first run
    print ("First call: ", jit(impure_uses_globals)(4.))
    g = 10.  # Update the global

    # Subsequent runs may silently use the cached value of the globals
    print ("Second call: ", jit(impure_uses_globals)(5.))

    # JAX re-runs the Python function when the type or shape of the argument changes
    # This will end up reading the latest value of the global
    print ("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))

    First call:  4.0
    Second call:  5.0
    Third call, different type:  [14.]

    g = 0.
    def impure_saves_global(x):
      global g
      g = x
      return x

    # JAX runs once the transformed function with special Traced values for arguments
    print ("First call: ", jit(impure_saves_global)(4.))
    print ("Saved global: ", g)  # Saved global has an internal JAX value

    First call:  4.0
    Saved global:  JitTracer(~float32[])

A Python function can be functionally pure even if it actually uses stateful objects internally, as long as it does not read or write external state:

    def pure_uses_internal_state(x):
      state = dict(even=0, odd=0)
      for i in range(10):
        state['even' if i % 2 == 0 else 'odd'] += x
      return state['even'] + state['odd']

    print(jit(pure_uses_internal_state)(5.))

    50.0

It is not recommended to use iterators in any JAX function you want to `jit` or in any control-flow primitive. The reason is that an iterator is a python object which introduces state to retrieve the next element. Therefore, it is incompatible with JAX’s functional programming model. In the code below, there are some examples of incorrect attempts to use iterators with JAX. Most of them return an error, but some give unexpected results.

    import jax.numpy as jnp
    from jax import make_jaxpr

    # lax.fori_loop
    array = jnp.arange(10)
    print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)) # expected result 45
    iterator = iter(range(10))
    print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0)) # unexpected result 0

    # lax.scan
    def func11(arr, extra):
        ones = jnp.ones(arr.shape)
        def body(carry, aelems):
            ae1, ae2 = aelems
            return (carry + ae1 * ae2 + extra, carry)
        return lax.scan(body, 0., (arr, ones))
    make_jaxpr(func11)(jnp.arange(16), 5.)
    # make_jaxpr(func11)(iter(range(16)), 5.) # throws error

    # lax.cond
    array_operand = jnp.array([0.])
    lax.cond(True, lambda x: x+1, lambda x: x-1, array_operand)
    iter_operand = iter(range(10))
    # lax.cond(True, lambda x: next(x)+1, lambda x: next(x)-1, iter_operand) # throws error

    45
    0

## 🔪 In-place updates[\#](#in-place-updates "Link to this heading")

In Numpy you’re used to doing this:

    numpy_array = np.zeros((3,3), dtype=np.float32)
    print("original array:")
    print(numpy_array)

    # In place, mutating update
    numpy_array[1, :] = 1.0
    print("updated array:")
    print(numpy_array)

    original array:
    [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]
    updated array:
    [[0. 0. 0.]
     [1. 1. 1.]
     [0. 0. 0.]]

If we try to do in-place indexed updating on a `jax.Array`, however, we get an **error**! (☉\_☉)

    %xmode Minimal

    Exception reporting mode: Minimal

    jax_array = jnp.zeros((3,3), dtype=jnp.float32)

    # In place update of JAX's array will yield an error!
    jax_array[1, :] = 1.0

    TypeError: JAX arrays are immutable and do not support in-place item assignment. Instead of x[idx] = y, use x = x.at[idx].set(y) or another .at[] method: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html

And if we try to do `__iadd__`-style in-place updating, we get **different behavior than NumPy**! (☉\_☉) (☉\_☉)

    jax_array = jnp.array([10, 20])
    jax_array_new = jax_array
    jax_array_new += 10
    print(jax_array_new)  # `jax_array_new` is rebound to a new value [20, 30], but...
    print(jax_array)      # the original value is unmodified as [10, 20] !

    numpy_array = np.array([10, 20])
    numpy_array_new = numpy_array
    numpy_array_new += 10
    print(numpy_array_new)  # `numpy_array_new is numpy_array`, and it was updated
    print(numpy_array)      # in-place, so both are [20, 30] !

    [20 30]
    [10 20]
    [20 30]
    [20 30]

That’s because NumPy defines `__iadd__` to perform in-place mutation. In contrast, `jax.Array` doesn’t define an `__iadd__`, so Python treats `jax_array_new`` ``+=`` ``10` as syntactic sugar for `jax_array_new`` ``=`` ``jax_array_new`` ``+`` ``10`, rebinding the variable without mutating any arrays.

Allowing mutation of variables in-place makes program analysis and transformation difficult. JAX requires that programs are pure functions.

Instead, JAX offers a *functional* array update using the [`.at` property on JAX arrays](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).

️⚠️ inside `jit`’d code and `lax.while_loop` or `lax.fori_loop` the **size** of slices can’t be functions of argument *values* but only functions of argument *shapes* – the slice start indices have no such restriction. See the below **Control Flow** Section for more information on this limitation.

### Array updates: `x.at[idx].set(y)`[\#](#array-updates-x-at-idx-set-y "Link to this heading")

For example, the update above can be written as:

    jax_array = jnp.zeros((3,3), dtype=jnp.float32)
    updated_array = jax_array.at[1, :].set(1.0)
    print("updated array:\n", updated_array)

    updated array:
     [[0. 0. 0.]
     [1. 1. 1.]
     [0. 0. 0.]]

JAX’s array update functions, unlike their NumPy versions, operate out-of-place. That is, the updated array is returned as a new array and the original array is not modified by the update.

    print("original array unchanged:\n", jax_array)

    original array unchanged:
     [[0. 0. 0.]
     [0. 0. 0.]
     [0. 0. 0.]]

However, inside **jit**-compiled code, if the **input value** `x` of `x.at[idx].set(y)` is not reused, the compiler will optimize the array update to occur *in-place*.

### Array updates with other operations[\#](#array-updates-with-other-operations "Link to this heading")

Indexed array updates are not limited simply to overwriting values. For example, we can perform indexed addition as follows:

    print("original array:")
    jax_array = jnp.ones((5, 6))
    print(jax_array)

    new_jax_array = jax_array.at[::2, 3:].add(7.)
    print("new array post-addition:")
    print(new_jax_array)

    original array:
    [[1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 1. 1. 1.]]
    new array post-addition:
    [[1. 1. 1. 8. 8. 8.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 8. 8. 8.]
     [1. 1. 1. 1. 1. 1.]
     [1. 1. 1. 8. 8. 8.]]

For more details on indexed array updates, see the [documentation for the `.at` property](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at).

## 🔪 Using `jax.jit` with class methods[\#](#using-jax-jit-with-class-methods "Link to this heading")

Most examples of [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) concern decorating stand-alone Python functions, but decorating a method within a class introduces some complication. For example, consider the following simple class, where we’ve used a standard `jax.jit` annotation on a method:

    import jax.numpy as jnp
    from jax import jit

    class CustomClass:
      def __init__(self, x: jnp.ndarray, mul: bool):
        self.x = x
        self.mul = mul

      @jit  # <---- How to do this correctly?
      def calc(self, y):
        if self.mul:
          return self.x * y
        return y

However, this approach will result in an error when you attempt to call this method:

    c = CustomClass(2, True)
    c.calc(3)

    TypeError: Error interpreting argument to <function CustomClass.calc at 0x70ceabbc1620> as an abstract array. The problematic value is of type <class '__main__.CustomClass'> and was passed to the function at path self.
    This typically means that a jit-wrapped function was called with a non-array argument, and this argument was not marked as static using the static_argnums or static_argnames parameters of jax.jit.

The problem is that the first argument to the function is `self`, which has type `CustomClass`, and JAX does not know how to handle this type. There are three basic strategies we might use in this case, and we’ll discuss them below.

### Strategy 1: JIT-compiled helper function[\#](#strategy-1-jit-compiled-helper-function "Link to this heading")

The most straightforward approach is to create a helper function external to the class that can be JIT-decorated in the normal way. For example:

    class CustomClass:
      def __init__(self, x: jnp.ndarray, mul: bool):
        self.x = x
        self.mul = mul

      def calc(self, y):
        return _calc(self.mul, self.x, y)

    @jit(static_argnums=0)
    def _calc(mul, x, y):
      if mul:
        return x * y
      return y

The result will work as expected:

    c = CustomClass(2, True)
    print(c.calc(3))

    6

The benefit of such an approach is that it is simple, explicit, and it avoids the need to teach JAX how to handle objects of type `CustomClass`. However, you may wish to keep all the method logic in the same place.

### Strategy 2: Marking `self` as static[\#](#strategy-2-marking-self-as-static "Link to this heading")

Another common pattern is to use `static_argnums` to mark the `self` argument as static. But this must be done with care to avoid unexpected results. You may be tempted to simply do this:

    class CustomClass:
      def __init__(self, x: jnp.ndarray, mul: bool):
        self.x = x
        self.mul = mul

      # WARNING: this example is broken, as we'll see below. Don't copy & paste!
      @jit(static_argnums=0)
      def calc(self, y):
        if self.mul:
          return self.x * y
        return y

If you call the method, it will no longer raise an error:

    c = CustomClass(2, True)
    print(c.calc(3))

    6

However, there is a catch: if you mutate the object after the first method call, the subsequent method call may return an incorrect result:

    c.mul = False
    print(c.calc(3))  # Should print 3

    6

Why is this? When you mark an object as static, it will effectively be used as a dictionary key in JIT’s internal compilation cache, meaning its hash (i.e. `hash(obj)`) equality (i.e. `obj1`` ``==`` ``obj2`) and object identity (i.e. `obj1`` ``is`` ``obj2`) will be assumed to have consistent behavior. The default `__hash__` for a custom object is its object ID, and so JAX has no way of knowing that a mutated object should trigger a re-compilation.

You can partially address this by defining an appropriate `__hash__` and `__eq__` methods for your object; for example:

    class CustomClass:
      def __init__(self, x: jnp.ndarray, mul: bool):
        self.x = x
        self.mul = mul

      @jit(static_argnums=0)
      def calc(self, y):
        if self.mul:
          return self.x * y
        return y

      def __hash__(self):
        return hash((self.x, self.mul))

      def __eq__(self, other):
        return (isinstance(other, CustomClass) and
                (self.x, self.mul) == (other.x, other.mul))

(see the [`object.__hash__`](https://docs.python.org/3/reference/datamodel.html#object.__hash__) documentation for more discussion of the requirements when overriding `__hash__`).

This should work correctly with JIT and other transforms **so long as you never mutate your object**. Mutations of objects used as hash keys lead to several subtle problems, which is why for example mutable Python containers (e.g. [`dict`](https://docs.python.org/3/library/stdtypes.html#dict), [`list`](https://docs.python.org/3/library/stdtypes.html#list)) don’t define `__hash__`, while their immutable counterparts (e.g. [`tuple`](https://docs.python.org/3/library/stdtypes.html#tuple)) do.

If your class relies on in-place mutations (such as setting `self.attr`` ``=`` ``...` within its methods), then your object is not really “static” and marking it as such may lead to problems. Fortunately, there’s another option for this case.

### Strategy 3: Making `CustomClass` a PyTree[\#](#strategy-3-making-customclass-a-pytree "Link to this heading")

The most flexible approach to correctly JIT-compiling a class method is to register the type as a custom PyTree object; see [Custom pytree nodes](https://docs.jax.dev/en/latest/custom_pytrees.html#pytrees-custom-pytree-nodes). This lets you specify exactly which components of the class should be treated as static and which should be treated as dynamic. Here’s how it might look:

    class CustomClass:
      def __init__(self, x: jnp.ndarray, mul: bool):
        self.x = x
        self.mul = mul

      @jit
      def calc(self, y):
        if self.mul:
          return self.x * y
        return y

      def _tree_flatten(self):
        children = (self.x,)  # arrays / dynamic values
        aux_data = {'mul': self.mul}  # static values
        return (children, aux_data)

      @classmethod
      def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    from jax import tree_util
    tree_util.register_pytree_node(CustomClass,
                                   CustomClass._tree_flatten,
                                   CustomClass._tree_unflatten)

This is certainly more involved, but it solves all the issues associated with the simpler approaches used above:

    c = CustomClass(2, True)
    print(c.calc(3))

    6

    c.mul = False  # mutation is detected
    print(c.calc(3))

    3

    c = CustomClass(jnp.array(2), True)  # non-hashable x is supported
    print(c.calc(3))

    6

So long as your `tree_flatten` and `tree_unflatten` functions correctly handle all relevant attributes in the class, you should be able to use objects of this type directly as arguments to JIT-compiled functions, without any special annotations.

## 🔪 Out-of-bounds indexing[\#](#out-of-bounds-indexing "Link to this heading")

In Numpy, you are used to errors being thrown when you index an array outside of its bounds, like this:

    np.arange(10)[11]

    IndexError: index 11 is out of bounds for axis 0 with size 10

However, raising an error from code running on an accelerator can be difficult or impossible. Therefore, JAX must choose some non-error behavior for out of bounds indexing (akin to how invalid floating point arithmetic results in `NaN`). When the indexing operation is an array index update (e.g. `index_add` or `scatter`-like primitives), updates at out-of-bounds indices will be skipped; when the operation is an array index retrieval (e.g. NumPy indexing or `gather`-like primitives) the index is clamped to the bounds of the array since **something** must be returned. For example, the last value of the array will be returned from this indexing operation:

    jnp.arange(10)[11]

    Array(9, dtype=int32)

If you would like finer-grained control over the behavior for out-of-bound indices, you can use the optional parameters of [`ndarray.at`](https://docs.jax.dev/en/latest/_autosummary/jax.numpy.ndarray.at.html); for example:

    jnp.arange(10.0).at[11].get()

    Array(9., dtype=float32)

    jnp.arange(10.0).at[11].get(mode='fill', fill_value=jnp.nan)

    Array(nan, dtype=float32)

Note that due to this behavior for index retrieval, functions like `jnp.nanargmin` and `jnp.nanargmax` return -1 for slices consisting of NaNs whereas Numpy would throw an error.

Note also that, as the two behaviors described above are not inverses of each other, reverse-mode automatic differentiation (which turns index updates into index retrievals and vice versa) [will not preserve the semantics of out of bounds indexing](https://github.com/jax-ml/jax/issues/5760). Thus it may be a good idea to think of out-of-bounds indexing in JAX as a case of [undefined behavior](https://en.wikipedia.org/wiki/Undefined_behavior).

## 🔪 Non-array inputs: NumPy vs. JAX[\#](#non-array-inputs-numpy-vs-jax "Link to this heading")

NumPy is generally happy accepting Python lists or tuples as inputs to its API functions:

    np.sum([1, 2, 3])

    np.int64(6)

JAX departs from this, generally returning a helpful error:

    jnp.sum([1, 2, 3])

    TypeError: sum requires ndarray or scalar arguments, got <class 'list'> at position 0.

This is a deliberate design choice, because passing lists or tuples to traced functions can lead to silent performance degradation that might otherwise be difficult to detect.

For example, consider the following permissive version of `jnp.sum` that allows list inputs:

    def permissive_sum(x):
      return jnp.sum(jnp.array(x))

    x = list(range(10))
    permissive_sum(x)

    Array(45, dtype=int32)

The output is what we would expect, but this hides potential performance issues under the hood. In JAX’s tracing and JIT compilation model, each element in a Python list or tuple is treated as a separate JAX variable, and individually processed and pushed to device. This can be seen in the jaxpr for the `permissive_sum` function above:

    make_jaxpr(permissive_sum)(x)

    { lambda ; a:i32[] b:i32[] c:i32[] d:i32[] e:i32[] f:i32[] g:i32[] h:i32[] i:i32[]
        j:i32[]. let
        k:i32[] = convert_element_type[new_dtype=int32 weak_type=False] a
        l:i32[1] = broadcast_in_dim k
        m:i32[] = convert_element_type[new_dtype=int32 weak_type=False] b
        n:i32[1] = broadcast_in_dim m
        o:i32[] = convert_element_type[new_dtype=int32 weak_type=False] c
        p:i32[1] = broadcast_in_dim o
        q:i32[] = convert_element_type[new_dtype=int32 weak_type=False] d
        r:i32[1] = broadcast_in_dim q
        s:i32[] = convert_element_type[new_dtype=int32 weak_type=False] e
        t:i32[1] = broadcast_in_dim s
        u:i32[] = convert_element_type[new_dtype=int32 weak_type=False] f
        v:i32[1] = broadcast_in_dim u
        w:i32[] = convert_element_type[new_dtype=int32 weak_type=False] g
        x:i32[1] = broadcast_in_dim w
        y:i32[] = convert_element_type[new_dtype=int32 weak_type=False] h
        z:i32[1] = broadcast_in_dim y
        ba:i32[] = convert_element_type[new_dtype=int32 weak_type=False] i
        bb:i32[1] = broadcast_in_dim ba
        bc:i32[] = convert_element_type[new_dtype=int32 weak_type=False] j
        bd:i32[1] = broadcast_in_dim bc
        be:i32[10] = concatenate[dimension=0] l n p r t v x z bb bd
        bf:i32[] = reduce_sum[axes=(0,) out_sharding=None] be
      in (bf,) }

Each entry of the list is handled as a separate input, resulting in a tracing & compilation overhead that grows linearly with the size of the list. To prevent surprises like this, JAX avoids implicit conversions of lists and tuples to arrays.

If you would like to pass a tuple or list to a JAX function, you can do so by first explicitly converting it to an array:

    jnp.sum(jnp.array(x))

    Array(45, dtype=int32)

## 🔪 Random numbers[\#](#random-numbers "Link to this heading")

JAX’s pseudo-random number generation differs from Numpy’s in important ways. For a quick how-to, see [Pseudorandom numbers](thinking_in_jax.html#key-concepts-prngs). For more details, see the [Pseudorandom numbers](../random-numbers.html#pseudorandom-numbers) tutorial.

## 🔪 Control flow[\#](#control-flow "Link to this heading")

Moved to [Control flow and logical operators with JIT](../control-flow.html#control-flow).

## 🔪 Dynamic shapes[\#](#dynamic-shapes "Link to this heading")

JAX code used within transforms like `jax.jit`, `jax.vmap`, `jax.grad`, etc. requires all output arrays and intermediate arrays to have static shape: that is, the shape cannot depend on values within other arrays.

For example, if you were implementing your own version of `jnp.nansum`, you might start with something like this:

    def nansum(x):
      mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
      x_without_nans = x[mask]
      return x_without_nans.sum()

Outside JIT and other transforms, this works as expected:

    x = jnp.array([1, 2, jnp.nan, 3, 4])
    print(nansum(x))

    10.0

If you attempt to apply `jax.jit` or another transform to this function, it will error:

    jax.jit(nansum)(x)

    NonConcreteBooleanIndexError: Array boolean indices must be concrete; got bool[5]

    See https://docs.jax.dev/en/latest/errors.html#jax.errors.NonConcreteBooleanIndexError

The problem is that the size of `x_without_nans` is dependent on the values within `x`, which is another way of saying its size is *dynamic*. Often in JAX it is possible to work-around the need for dynamically-sized arrays via other means. For example, here it is possible to use the three-argument form of `jnp.where` to replace the NaN values with zeros, thus computing the same result while avoiding dynamic shapes:

    @jax.jit
    def nansum_2(x):
      mask = ~jnp.isnan(x)  # boolean mask selecting non-nan values
      return jnp.where(mask, x, 0).sum()

    print(nansum_2(x))

    10.0

Similar tricks can be played in other situations where dynamically-shaped arrays occur.

## 🔪 Debugging NaNs and Infs[\#](#debugging-nans-and-infs "Link to this heading")

Use the `jax_debug_nans` and `jax_debug_infs` flags to find the source of NaN/Inf values in functions and gradients. See [JAX debugging flags](../debugging/flags.html#debugging-flags).

## 🔪 Double (64bit) precision[\#](#double-64bit-precision "Link to this heading")

At the moment, JAX by default enforces single-precision numbers to mitigate the Numpy API’s tendency to aggressively promote operands to `double`. This is the desired behavior for many machine-learning applications, but it may catch you by surprise!

    x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)
    x.dtype

    /tmp/ipykernel_2205/1258726447.py:1: UserWarning: Explicitly requested dtype float64 is not available, and will be truncated to dtype float32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
      x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)

    dtype('float32')

To use double-precision numbers, you need to set the `jax_enable_x64` configuration variable **at startup**.

There are a few ways to do this:

1.  You can enable 64-bit mode by setting the environment variable `JAX_ENABLE_X64=True`.

2.  You can manually set the `jax_enable_x64` configuration flag at startup:

        # again, this only works on startup!
        import jax
        jax.config.update("jax_enable_x64", True)

3.  You can parse command-line flags with `absl.app.run(main)`

        import jax
        jax.config.config_with_absl()

4.  If you want JAX to run absl parsing for you, i.e. you don’t want to do `absl.app.run(main)`, you can instead use

        import jax
        if __name__ == '__main__':
          # calls jax.config.config_with_absl() *and* runs absl parsing
          jax.config.parse_flags_with_absl()

Note that \#2-#4 work for *any* of JAX’s configuration options.

We can then confirm that `x64` mode is enabled, for example:

    import jax
    import jax.numpy as jnp
    from jax import random

    jax.config.update("jax_enable_x64", True)
    x = random.uniform(random.key(0), (1000,), dtype=jnp.float64)
    x.dtype # --> dtype('float64')

### Caveats[\#](#caveats "Link to this heading")

⚠️ XLA doesn’t support 64-bit convolutions on all backends!

## 🔪 Miscellaneous divergences from NumPy[\#](#miscellaneous-divergences-from-numpy "Link to this heading")

While `jax.numpy` makes every attempt to replicate the behavior of numpy’s API, there do exist corner cases where the behaviors differ. Many such cases are discussed in detail in the sections above; here we list several other known places where the APIs diverge.

- For binary operations, JAX’s type promotion rules differ somewhat from those used by NumPy. See [Type Promotion Semantics](https://docs.jax.dev/en/latest/type_promotion.html) for more details.

- When performing unsafe type casts (i.e. casts in which the target dtype cannot represent the input value), JAX’s behavior may be backend dependent, and in general may diverge from NumPy’s behavior. Numpy allows control over the result in these scenarios via the `casting` argument (see [`np.ndarray.astype`](https://numpy.org/devdocs/reference/generated/numpy.ndarray.astype.html)); JAX does not provide any such configuration, instead directly inheriting the behavior of [XLA:ConvertElementType](https://www.openxla.org/xla/operation_semantics#convertelementtype).

  Here is an example of an unsafe cast with differing results between NumPy and JAX:

      >>> np.arange(254.0, 258.0).astype('uint8')
      array([254, 255,   0,   1], dtype=uint8)

      >>> jnp.arange(254.0, 258.0).astype('uint8')
      Array([254, 255, 255, 255], dtype=uint8)

  This sort of mismatch would typically arise when casting extreme values from floating to integer types or vice versa.

- When operating on [subnormal](https://en.wikipedia.org/wiki/Subnormal_number) floating point numbers, JAX operations use flush-to-zero semantics on some backends. For example:

      >>> import jax.numpy as jnp
      >>> subnormal = jnp.float32(1E-45)
      >>> subnormal  # subnormals are representable
      Array(1.e-45, dtype=float32)
      >>> subnormal + 0  # but are flushed to zero within operations
      Array(0., dtype=float32)

  The detailed operation semantics for subnormal values will generally vary depending on the backend.

## 🔪 Sharp bits covered in tutorials[\#](#sharp-bits-covered-in-tutorials "Link to this heading")

- [Control flow and logical operators with JIT](../control-flow.html#control-flow) discusses how to work with the constraints that `jit` imposes on the use of Python control flow and logical operators.

- [Stateful computations](../stateful-computations.html#stateful-computations) gives some advice on how to properly handle state in a JAX program, given that JAX transformations can be applied only to pure functions.

## Fin.[\#](#fin "Link to this heading")

If something’s not covered here that has caused you weeping and gnashing of teeth, please let us know and we’ll extend these introductory *advisos*!

[](thinking_in_jax.html "previous page")

previous

Quickstart: How to think in JAX

[](../jax-101.html "next page")

next

JAX 101

Contents

- [🔪 Pure functions](#pure-functions)
- [🔪 In-place updates](#in-place-updates)
  - [Array updates: `x.at[idx].set(y)`](#array-updates-x-at-idx-set-y)
  - [Array updates with other operations](#array-updates-with-other-operations)
- [🔪 Using `jax.jit` with class methods](#using-jax-jit-with-class-methods)
  - [Strategy 1: JIT-compiled helper function](#strategy-1-jit-compiled-helper-function)
  - [Strategy 2: Marking `self` as static](#strategy-2-marking-self-as-static)
  - [Strategy 3: Making `CustomClass` a PyTree](#strategy-3-making-customclass-a-pytree)
- [🔪 Out-of-bounds indexing](#out-of-bounds-indexing)
- [🔪 Non-array inputs: NumPy vs. JAX](#non-array-inputs-numpy-vs-jax)
- [🔪 Random numbers](#random-numbers)
- [🔪 Control flow](#control-flow)
- [🔪 Dynamic shapes](#dynamic-shapes)
- [🔪 Debugging NaNs and Infs](#debugging-nans-and-infs)
- [🔪 Double (64bit) precision](#double-64bit-precision)
  - [Caveats](#caveats)
- [🔪 Miscellaneous divergences from NumPy](#miscellaneous-divergences-from-numpy)
- [🔪 Sharp bits covered in tutorials](#sharp-bits-covered-in-tutorials)
- [Fin.](#fin)

By The JAX authors

© Copyright 2024, The JAX Authors.\
