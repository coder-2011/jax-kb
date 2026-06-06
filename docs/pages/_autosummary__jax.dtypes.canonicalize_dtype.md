- [](../index.html)
- [API Reference](../jax.html)
- [`jax.dtypes` module](../jax.dtypes.html)
- jax.dtypes.canonicalize_dtype

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.dtypes.canonicalize_dtype.rst "Download source file")
-  .pdf

# jax.dtypes.canonicalize_dtype

## Contents

- [`canonicalize_dtype()`](#jax.dtypes.canonicalize_dtype)

# jax.dtypes.canonicalize_dtype[\#](#jax-dtypes-canonicalize-dtype "Link to this heading")

jax.dtypes.canonicalize_dtype(*dtype*, *allow_extended_dtype=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/dtypes.py#L383-L387)[\#](#jax.dtypes.canonicalize_dtype "Link to this definition")  
Convert from a dtype to a canonical dtype based on config.x64_enabled.

Parameters:  
- **dtype** (*Any*)

- **allow_extended_dtype** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
DType \| ExtendedDType

[](jax.dtypes.bfloat16.html "previous page")

previous

jax.dtypes.bfloat16

[](jax.dtypes.float0.html "next page")

next

jax.dtypes.float0

Contents

- [`canonicalize_dtype()`](#jax.dtypes.canonicalize_dtype)

By The JAX authors

© Copyright 2024, The JAX Authors.\
