- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.optimization_barrier

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.optimization_barrier.rst "Download source file")
-  .pdf

# jax.lax.optimization_barrier

## Contents

- [`optimization_barrier()`](#jax.lax.optimization_barrier)

# jax.lax.optimization_barrier[\#](#jax-lax-optimization-barrier "Link to this heading")

jax.lax.optimization_barrier(*operand*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L9695-L9737)[\#](#jax.lax.optimization_barrier "Link to this definition")  
Prevents the compiler from moving operations across the barrier.

Optimization barriers have a number of possible uses:

- An optimization barrier ensures that every output of the barrier that is used by any operator, has been evaluated before any operator that depends on one of the barrier’s outputs. This can be used to enforce a particular order of operations.

  Note that all operands must be used through the barrier for this to work. There are no ordering constraints between an operator that uses one of the barrier’s outputs, and an operator that directly (not through the barrier) uses one of the barrier’s inputs.

- An optimization barrier prevents common subexpression elimination. This is used by JAX to implement rematerialization.

- Optimization barriers prevent compiler fusions. That is, operations before the barrier may not be fused into the same kernel as operations after the barrier by the compiler.

JAX does not define derivative or batching rules for an optimization barrier.

Optimization barriers have no effect outside a compiled function.

Parameters:  
**operand** – a pytree of JAX values.

Returns:  
A pytree of JAX values, with the same structure and contents as `operand`.

Examples

Prevents common-subexpression elimination between the two calls to sin:

    >>> def f(x):
    ...   return jax.lax.optimization_barrier(jax.lax.sin(x)) + jax.lax.sin(x)
    >>> jax.jit(f)(0.)
    Array(0., dtype=float32, weak_type=True)

[](jax.lax.nextafter.html "previous page")

previous

jax.lax.nextafter

[](jax.lax.pad.html "next page")

next

jax.lax.pad

Contents

- [`optimization_barrier()`](#jax.lax.optimization_barrier)

By The JAX authors

© Copyright 2024, The JAX Authors.\
