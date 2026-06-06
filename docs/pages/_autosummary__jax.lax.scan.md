- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.scan

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.scan.rst "Download source file")
-  .pdf

# jax.lax.scan

## Contents

- [`scan()`](#jax.lax.scan)

# jax.lax.scan[\#](#jax-lax-scan "Link to this heading")

jax.lax.scan(*f*, *init*, *xs=None*, *length=None*, *reverse=False*, *unroll=1*, *\_split_transpose=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/loops.py#L251-L481)[\#](#jax.lax.scan "Link to this definition")  
Scan a function over leading array axes while carrying along state.

The [Haskell-like type signature](https://wiki.haskell.org/Type_signature) in brief is

    scan :: (c -> a -> (c, b)) -> c -> [a] -> (c, [b])

where for any array type specifier `t`, `[t]` represents the type with an additional leading axis, and if `t` is a pytree (container) type with array leaves then `[t]` represents the type with the same pytree structure and corresponding leaves each with an additional leading axis.

When the type of `xs` (denoted a above) is an array type or None, and the type of `ys` (denoted b above) is an array type, the semantics of [`scan()`](#jax.lax.scan "jax.lax.scan") are given roughly by this Python implementation:

    def scan(f, init, xs, length=None):
      if xs is None:
        xs = [None] * length
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, np.stack(ys)

Unlike that Python version, both `xs` and `ys` may be arbitrary pytree values, and so multiple arrays can be scanned over at once and produce multiple output arrays. `None` is actually a special case of this, as it represents an empty pytree.

Also unlike that Python version, [`scan()`](#jax.lax.scan "jax.lax.scan") is a JAX primitive and is lowered to a single WhileOp. That makes it useful for reducing compilation times for JIT-compiled functions, since native Python loop constructs in an [`jit()`](jax.jit.html#jax.jit "jax.jit") function are unrolled, leading to large XLA computations.

Finally, the loop-carried value `carry` must hold a fixed shape and dtype across all iterations (and not just be consistent up to NumPy rank/shape broadcasting and dtype promotion rules, for example). In other words, the type `c` in the type signature above represents an array with a fixed shape and dtype (or a nested tuple/list/dict container data structure with a fixed structure and arrays with fixed shape and dtype at the leaves).

Note

[`scan()`](#jax.lax.scan "jax.lax.scan") compiles `f`, so while it can be combined with `jit()`, it’s usually unnecessary.

Note

[`scan()`](#jax.lax.scan "jax.lax.scan") is designed for iterating with a static number of iterations. For iteration with a dynamic number of iterations, use [`fori_loop()`](jax.lax.fori_loop.html#jax.lax.fori_loop "jax.lax.fori_loop") or [`while_loop()`](jax.lax.while_loop.html#jax.lax.while_loop "jax.lax.while_loop").

Parameters:  
- **f** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[Carry,* *X\],* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Carry,* *Y\]\]*) – a Python function to be scanned of type `c`` ``->`` ``a`` ``->`` ``(c,`` ``b)`, meaning that `f` accepts two arguments where the first is a value of the loop carry and the second is a slice of `xs` along its leading axis, and that `f` returns a pair where the first element represents a new value for the loop carry and the second represents a slice of the output.

- **init** (*Carry*) – an initial loop carry value of type `c`, which can be a scalar, array, or any pytree (nested Python tuple/list/dict) thereof, representing the initial loop carry value. This value must have the same structure as the first element of the pair returned by `f`.

- **xs** (*X* *\|* *None*) – the value of type `[a]` over which to scan along the leading axis, where `[a]` can be an array or any pytree (nested Python tuple/list/dict) thereof with consistent leading axis sizes.

- **length** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – optional integer specifying the number of loop iterations, which must agree with the sizes of leading axes of the arrays in `xs` (but can be used to perform scans where no input `xs` are needed).

- **reverse** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – optional boolean specifying whether to run the scan iteration forward (the default) or in reverse, equivalent to reversing the leading axes of the arrays in both `xs` and in `ys`.

- **unroll** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – optional non-negative int or bool specifying, in the underlying operation of the scan primitive, how many scan iterations to unroll within a single iteration of a loop. If an integer is provided, it determines how many unrolled loop iterations to run within a single rolled iteration of the loop. unroll=0 unrolls the entire loop. If a boolean is provided, it will determine if the loop is completely unrolled (i.e. unroll=True) or left completely rolled (i.e. unroll=False).

- **\_split_transpose** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Returns:  
A pair of type `(c,`` ``[b])` where the first element represents the final loop carry value and the second element represents the stacked outputs of the second output of `f` when scanned over the leading axis of the inputs.

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[Carry, Y\]

[](jax.lax.map.html "previous page")

previous

jax.lax.map

[](jax.lax.select.html "next page")

next

jax.lax.select

Contents

- [`scan()`](#jax.lax.scan)

By The JAX authors

© Copyright 2024, The JAX Authors.\
