- [](../index.html)
- [API Reference](../jax.html)
- [`jax.dtypes` module](../jax.dtypes.html)
- jax.dtypes.issubdtype

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.dtypes.issubdtype.rst "Download source file")
-  .pdf

# jax.dtypes.issubdtype

## Contents

- [`issubdtype()`](#jax.dtypes.issubdtype)

# jax.dtypes.issubdtype[\#](#jax-dtypes-issubdtype "Link to this heading")

jax.dtypes.issubdtype(*a*, *b*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/dtypes.py#L528-L551)[\#](#jax.dtypes.issubdtype "Link to this definition")  
Returns True if first argument is a typecode lower/equal in type hierarchy.

This is like [`numpy.issubdtype()`](https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html#numpy.issubdtype "(in NumPy v2.4)"), but can handle dtype extensions such as [`jax.dtypes.bfloat16`](jax.dtypes.bfloat16.html#jax.dtypes.bfloat16 "jax.dtypes.bfloat16") and jax.dtypes.prng_key.

Parameters:  
- **a** (*DTypeLike* *\|* *ExtendedDType* *\|* *None*)

- **b** (*DTypeLike* *\|* *ExtendedDType* *\|* *None*)

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

[](jax.dtypes.itemsize_bits.html "previous page")

previous

jax.dtypes.itemsize_bits

[](jax.dtypes.prng_key.html "next page")

next

jax.dtypes.prng_key

Contents

- [`issubdtype()`](#jax.dtypes.issubdtype)

By The JAX authors

© Copyright 2024, The JAX Authors.\
