- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.multiple_of

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.multiple_of.rst "Download source file")
-  .pdf

# jax.experimental.pallas.multiple_of

## Contents

- [`multiple_of()`](#jax.experimental.pallas.multiple_of)

# jax.experimental.pallas.multiple_of[\#](#jax-experimental-pallas-multiple-of "Link to this heading")

jax.experimental.pallas.multiple_of(*x*, *values*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L124-L139)[\#](#jax.experimental.pallas.multiple_of "Link to this definition")  
A compiler hint that asserts a value is a static multiple of another.

Note that misusing this function, such as asserting `x` is a multiple of `N` when it is not, can result in undefined behavior.

Parameters:  
- **x** (*jax_typing.Array*) – The input array.

- **values** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – A set of static divisors that `x` is a multiple of.

Returns:  
A copy of `x`.

Return type:  
jax_typing.Array

[](jax.experimental.pallas.loop.html "previous page")

previous

jax.experimental.pallas.loop

[](jax.experimental.pallas.run_scoped.html "next page")

next

jax.experimental.pallas.run_scoped

Contents

- [`multiple_of()`](#jax.experimental.pallas.multiple_of)

By The JAX authors

© Copyright 2024, The JAX Authors.\
