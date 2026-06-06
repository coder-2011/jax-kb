- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.transpose

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.transpose.rst "Download source file")
-  .pdf

# jax.lax.transpose

## Contents

- [`transpose()`](#jax.lax.transpose)

# jax.lax.transpose[\#](#jax-lax-transpose "Link to this heading")

jax.lax.transpose(*operand*, *permutation*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3155-L3166)[\#](#jax.lax.transpose "Link to this definition")  
Wraps XLA’s [Transpose](https://www.openxla.org/xla/operation_semantics#transpose) operator.

Parameters:  
- **operand** (*ArrayLike*)

- **permutation** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *np.ndarray*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.lax.top_k.html "previous page")

previous

jax.lax.top_k

[](jax.lax.unstack.html "next page")

next

jax.lax.unstack

Contents

- [`transpose()`](#jax.lax.transpose)

By The JAX authors

© Copyright 2024, The JAX Authors.\
