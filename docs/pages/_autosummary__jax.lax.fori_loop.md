- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.fori_loop

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.fori_loop.rst "Download source file")
-  .pdf

# jax.lax.fori_loop

## Contents

- [`fori_loop()`](#jax.lax.fori_loop)

# jax.lax.fori_loop[\#](#jax-lax-fori-loop "Link to this heading")

jax.lax.fori_loop(*lower*, *upper*, *body_fun*, *init_val*, *\**, *unroll=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py#L2470-L2600)[\#](#jax.lax.fori_loop "Link to this definition")  
Loop from `lower` to `upper` by reduction to [`jax.lax.while_loop()`](jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop").

The [Haskell-like type signature](https://wiki.haskell.org/Type_signature) in brief is

    fori_loop :: Int -> Int -> ((Int, a) -> a) -> a -> a

The semantics of `fori_loop` are given by this Python implementation:

    def fori_loop(lower, upper, body_fun, init_val):
      val = init_val
      for i in range(lower, upper):
        val = body_fun(i, val)
      return val

As the Python version suggests, setting `upper`` ``<=`` ``lower` will produce no iterations. Negative or custom increments are not supported.

Unlike that Python version, `fori_loop` is implemented in terms of either a call to [`jax.lax.while_loop()`](jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop") or a call to [`jax.lax.scan()`](jax.lax.scan.html#jax.lax.scan "jax.lax.scan"). If the trip count is static (meaning known at tracing time, perhaps because `lower` and `upper` are Python integer literals) then the `fori_loop` is implemented in terms of [`scan()`](jax.lax.scan.html#jax.lax.scan "jax.lax.scan") and reverse-mode autodiff is supported; otherwise, a `while_loop` is used and reverse-mode autodiff is not supported. See those functions’ docstrings for more information.

Also unlike the Python analogue, the loop-carried value `val` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `a` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

Note

[`fori_loop()`](#jax.lax.fori_loop "jax.lax.fori_loop") compiles `body_fun`, so while it can be combined with `jit()`, it’s usually unnecessary.

Parameters:  
- **lower** – an integer representing the loop index lower bound (inclusive)

- **upper** – an integer representing the loop index upper bound (exclusive)

- **body_fun** – function of type `(int,`` ``a)`` ``->`` ``a`.

- **init_val** – initial loop carry value of type `a`.

- **unroll** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* *None*) – An optional integer or boolean that determines how much to unroll the loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. If a boolean is provided, it will determine if the loop is completely unrolled (i.e. unroll=True) or left completely unrolled (i.e. unroll=False). This argument is only applicable if the loop bounds are statically known.

Returns:  
Loop value from the final iteration, of type `a`.

[](jax.lax.cond.html "previous page")

previous

jax.lax.cond

[](jax.lax.map.html "next page")

next

jax.lax.map

Contents

- [`fori_loop()`](#jax.lax.fori_loop)

By The JAX authors

© Copyright 2024, The JAX Authors.\
