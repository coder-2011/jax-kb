- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.sort

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.sort.rst "Download source file")
-  .pdf

# jax.lax.sort

## Contents

- [`sort()`](#jax.lax.sort)

# jax.lax.sort[\#](#jax-lax-sort "Link to this heading")

jax.lax.sort(*operand: [Array](jax.Array.html#jax.Array "jax.Array")*, *dimension: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = -1*, *is_stable: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *num_keys: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 1*) → [Array](jax.Array.html#jax.Array "jax.Array")[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3503-L3539)[\#](#jax.lax.sort "Link to this definition")\
jax.lax.sort(*operand: Sequence\[[Array](jax.Array.html#jax.Array "jax.Array")\]*, *dimension: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = -1*, *is_stable: [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") = True*, *num_keys: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") = 1*) → [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), ...\]  
Wraps XLA’s [Sort](https://www.openxla.org/xla/operation_semantics#sort) operator.

For floating point inputs, -0.0 and 0.0 are treated as equivalent, and NaN values are sorted to the end of the array. For complex inputs, the sort order is lexicographic over the real and imaginary parts, with the real part primary.

Parameters:  
- **operand** – Array or sequence of arrays

- **dimension** – integer dimension along which to sort. Default: -1.

- **is_stable** – boolean specifying whether to use a stable sort. Default: True.

- **num_keys** – number of operands to treat as sort keys. Default: 1. For num_keys \> 1, the sort order will be determined lexicographically using the first num_keys arrays, with the first key being primary. The remaining operands will be returned with the same permutation.

Returns:  
sorted version of the input or inputs.

Return type:  
operand

[](jax.lax.slice_in_dim.html "previous page")

previous

jax.lax.slice_in_dim

[](jax.lax.sort_key_val.html "next page")

next

jax.lax.sort_key_val

Contents

- [`sort()`](#jax.lax.sort)

By The JAX authors

© Copyright 2024, The JAX Authors.\
