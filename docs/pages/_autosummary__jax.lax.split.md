- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.split

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.split.rst "Download source file")
-  .pdf

# jax.lax.split

## Contents

- [`split()`](#jax.lax.split)

# jax.lax.split[\#](#jax-lax-split "Link to this heading")

jax.lax.split(*operand*, *sizes*, *axis=0*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2056-L2074)[\#](#jax.lax.split "Link to this definition")  
Splits an array along `axis`.

Parameters:  
- **operand** (*ArrayLike*) – an array to split

- **sizes** (*Sequence\[DimSize\]*) – the sizes of the split arrays. The sum of the sizes must be equal to the size of the `axis` dimension of `operand`.

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis along which to split the array.

Returns:  
A sequence of `len(sizes)` arrays. If `sizes` is `[s1,`` ``s2,`` ``...]`, this function returns chunks of sizes `s1`, `s2`, taken along `axis`.

Return type:  
Sequence\[[Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.sort_key_val.html "previous page")

previous

jax.lax.sort_key_val

[](jax.lax.sqrt.html "next page")

next

jax.lax.sqrt

Contents

- [`split()`](#jax.lax.split)

By The JAX authors

© Copyright 2024, The JAX Authors.\
