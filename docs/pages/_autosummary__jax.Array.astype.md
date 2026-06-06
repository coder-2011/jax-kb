- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.astype

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.astype.rst "Download source file")
-  .pdf

# jax.Array.astype

## Contents

- [`Array.astype()`](#jax.Array.astype)

# jax.Array.astype[\#](#jax-array-astype "Link to this heading")

*abstract* Array.astype(*dtype*, *copy=False*, *device=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_methods.py#L116-L126)[\#](#jax.Array.astype "Link to this definition")  
Copy the array and cast to a specified dtype.

This is implemented via [`jax.lax.convert_element_type()`](jax.lax.convert_element_type.html#jax.lax.convert_element_type "jax.lax.convert_element_type"), which may have slightly different behavior than [`numpy.ndarray.astype()`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype "(in NumPy v2.4)") in some cases. In particular, the details of float-to-int and int-to-float casts are implementation dependent.

Parameters:  
- **self** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **dtype** (*DTypeLike* *\|* *None*)

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

- **device** (*xc.Device* *\|* [*Sharding*](../jax.sharding.html#jax.sharding.Sharding "jax.sharding.Sharding") *\|* *None*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.Array.argsort.html "previous page")

previous

jax.Array.argsort

[](jax.Array.at.html "next page")

next

jax.Array.at

Contents

- [`Array.astype()`](#jax.Array.astype)

By The JAX authors

© Copyright 2024, The JAX Authors.\
