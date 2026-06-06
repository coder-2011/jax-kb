- [](index.html)
- [API Reference](jax.html)
- `jax.dtypes` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.dtypes.rst "Download source file")
-  .pdf

# jax.dtypes module

# `jax.dtypes` module[\#](#module-jax.dtypes "Link to this heading")

|  |  |
|----|----|
| [`bfloat16`](_autosummary/jax.dtypes.bfloat16.html#jax.dtypes.bfloat16 "jax.dtypes.bfloat16") | bfloat16 floating-point values |
| [`canonicalize_dtype`](_autosummary/jax.dtypes.canonicalize_dtype.html#jax.dtypes.canonicalize_dtype "jax.dtypes.canonicalize_dtype")(dtype\[, allow_extended_dtype\]) | Convert from a dtype to a canonical dtype based on config.x64_enabled. |
| [`float0`](_autosummary/jax.dtypes.float0.html#jax.dtypes.float0 "jax.dtypes.float0") | VoidDType(length, /) |
| [`itemsize_bits`](_autosummary/jax.dtypes.itemsize_bits.html#jax.dtypes.itemsize_bits "jax.dtypes.itemsize_bits")(dtype) | Number of bits per element for the dtype. |
| [`issubdtype`](_autosummary/jax.dtypes.issubdtype.html#jax.dtypes.issubdtype "jax.dtypes.issubdtype")(a, b) | Returns True if first argument is a typecode lower/equal in type hierarchy. |
| [`prng_key`](_autosummary/jax.dtypes.prng_key.html#jax.dtypes.prng_key "jax.dtypes.prng_key")() | Scalar class for PRNG Key dtypes. |
| [`result_type`](_autosummary/jax.dtypes.result_type.html#jax.dtypes.result_type "jax.dtypes.result_type")(\*args\[, return_weak_type_flag\]) | Convenience function to apply JAX argument dtype promotion. |
| [`scalar_type_of`](_autosummary/jax.dtypes.scalar_type_of.html#jax.dtypes.scalar_type_of "jax.dtypes.scalar_type_of")(x) | Return the scalar type associated with a JAX value. |
| [`TypePromotionError`](_autosummary/jax.dtypes.TypePromotionError.html#jax.dtypes.TypePromotionError "jax.dtypes.TypePromotionError") | Raised when JAX type promotion fails. |

[](_autosummary/jax.distributed.shutdown.html "previous page")

previous

jax.distributed.shutdown

[](_autosummary/jax.dtypes.bfloat16.html "next page")

next

jax.dtypes.bfloat16

By The JAX authors

© Copyright 2024, The JAX Authors.\
