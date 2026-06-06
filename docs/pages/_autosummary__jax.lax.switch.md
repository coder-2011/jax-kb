- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.switch

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.switch.rst "Download source file")
-  .pdf

# jax.lax.switch

## Contents

- [`switch()`](#jax.lax.switch)

# jax.lax.switch[\#](#jax-lax-switch "Link to this heading")

jax.lax.switch(*index*, *branches*, *\*operands*, *operand=\<object object\>*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/control_flow/conditionals.py#L65-L131)[\#](#jax.lax.switch "Link to this definition")  
Apply exactly one of the `branches` given by `index`.

If `index` is out of bounds, it is clamped to within bounds.

Has the semantics of the following Python:

    def switch(index, branches, *operands):
      index = clamp(0, index, len(branches) - 1)
      return branches[index](*operands)

Internally this wraps XLA’s [Conditional](https://www.openxla.org/xla/operation_semantics#conditional) operator. However, when transformed with [`vmap()`](jax.vmap.html#jax.vmap "jax.vmap") to operate over a batch of predicates, `cond` is converted to [`select()`](jax.lax.select.html#jax.lax.select "jax.lax.select").

Parameters:  
- **index** – Integer scalar type, indicating which branch function to apply.

- **branches** (*Sequence\[*[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\]*) – Sequence of functions (A -\> B) to be applied based on `index`. All branches must return the same output structure.

- **operands** (*Any*) – Operands (A) input to whichever branch is applied.

- **operand** (*Any*)

Returns:  
Value (B) of `branch(*operands)` for the branch that was selected based on `index`.

[](jax.lax.select_n.html "previous page")

previous

jax.lax.select_n

[](jax.lax.while_loop.html "next page")

next

jax.lax.while_loop

Contents

- [`switch()`](#jax.lax.switch)

By The JAX authors

© Copyright 2024, The JAX Authors.\
