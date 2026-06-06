- [](../index.html)
- [API Reference](../jax.html)
- [`jax.dtypes` module](../jax.dtypes.html)
- jax.dtypes.result_type

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.dtypes.result_type.rst "Download source file")
-  .pdf

# jax.dtypes.result_type

## Contents

- [`result_type()`](#jax.dtypes.result_type)

# jax.dtypes.result_type[\#](#jax-dtypes-result-type "Link to this heading")

jax.dtypes.result_type(*\*args*, *return_weak_type_flag=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/dtypes.py#L1115-L1133)[\#](#jax.dtypes.result_type "Link to this definition")  
Convenience function to apply JAX argument dtype promotion.

Parameters:  
- **return_weak_type_flag** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – if True, then return a `(dtype,`` ``weak_type)` tuple. If False, just return dtype

- **args** (*Any*)

Returns:  
dtype or (dtype, weak_type) depending on the value of the `return_weak_type` argument.

Return type:  
DType \| [tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[DType, [bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")\]

[](jax.dtypes.prng_key.html "previous page")

previous

jax.dtypes.prng_key

[](jax.dtypes.scalar_type_of.html "next page")

next

jax.dtypes.scalar_type_of

Contents

- [`result_type()`](#jax.dtypes.result_type)

By The JAX authors

© Copyright 2024, The JAX Authors.\
