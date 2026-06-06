- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.while_loop

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.while_loop.rst "Download source file")
-  .pdf

# jax.lax.while_loop

## Contents

- [`while_loop()`](#jax.lax.while_loop)

# jax.lax.while_loop[\#](#jax-lax-while-loop "Link to this heading")

jax.lax.while_loop(*cond_fun*, *body_fun*, *init_val*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py#L1542-L1691)[\#](#jax.lax.while_loop "Link to this definition")  
Call `body_fun` repeatedly in a loop while `cond_fun` is True.

The [Haskell-like type signature](https://wiki.haskell.org/Type_signature) in brief is

    while_loop :: (a -> Bool) -> (a -> a) -> a -> a

The semantics of `while_loop` are given by this Python implementation:

    def while_loop(cond_fun, body_fun, init_val):
      val = init_val
      while cond_fun(val):
        val = body_fun(val)
      return val

Unlike that Python version, `while_loop` is a JAX primitive and is lowered to a single WhileOp. That makes it useful for reducing compilation times for jit-compiled functions, since native Python loop constructs in an `@jit` function are unrolled, leading to large XLA computations.

Also unlike the Python analogue, the loop-carried value `val` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `a` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

Another difference from using Python-native loop constructs is that `while_loop` is not reverse-mode differentiable because XLA computations require static bounds on memory requirements.

Note

[`while_loop()`](#jax.lax.while_loop "jax.lax.while_loop") compiles `cond_fun` and `body_fun`, so while it can be combined with `jit()`, it’s usually unnecessary.

Parameters:  
- **cond_fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T\],* *BooleanNumeric\]*) – function of type `a`` ``->`` ``Bool`.

- **body_fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T\],* *T\]*) – function of type `a`` ``->`` ``a`.

- **init_val** (*T*) – value of type `a`, a type that can be a scalar, array, or any pytree (nested Python tuple/list/dict) thereof, representing the initial loop carry value.

Returns:  
The output from the final iteration of body_fun, of type `a`.

Return type:  
T

[](jax.lax.switch.html "previous page")

previous

jax.lax.switch

[](jax.lax.stop_gradient.html "next page")

next

jax.lax.stop_gradient

Contents

- [`while_loop()`](#jax.lax.while_loop)

By The JAX authors

© Copyright 2024, The JAX Authors.\
