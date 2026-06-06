- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.sort_key_val

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.sort_key_val.rst "Download source file")
-  .pdf

# jax.lax.sort_key_val

## Contents

- [`sort_key_val()`](#jax.lax.sort_key_val)

# jax.lax.sort_key_val[\#](#jax-lax-sort-key-val "Link to this heading")

jax.lax.sort_key_val(*keys*, *values*, *dimension=-1*, *is_stable=True*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L3540-L3546)[\#](#jax.lax.sort_key_val "Link to this definition")  
Sorts `keys` along `dimension` and applies the same permutation to `values`.

Parameters:  
- **keys** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **values** (*ArrayLike*)

- **dimension** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)"))

- **is_stable** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Return type:  
[tuple](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array"), [Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.lax.sort.html "previous page")

previous

jax.lax.sort

[](jax.lax.split.html "next page")

next

jax.lax.split

Contents

- [`sort_key_val()`](#jax.lax.sort_key_val)

By The JAX authors

© Copyright 2024, The JAX Authors.\
