- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.repeat

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.repeat.rst "Download source file")
-  .pdf

# jax.Array.repeat

## Contents

- [`Array.repeat()`](#jax.Array.repeat)

# jax.Array.repeat[\#](#jax-array-repeat "Link to this heading")

*abstract* Array.repeat(*repeats*, *axis=None*, *\**, *total_repeat_length=None*, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_methods.py#L357-L367)[\#](#jax.Array.repeat "Link to this definition")  
Construct an array from repeated elements.

Refer to [`jax.numpy.repeat()`](jax.numpy.repeat.html#jax.numpy.repeat "jax.numpy.repeat") for the full documentation.

Parameters:  
- **self** ([*Array*](jax.Array.html#jax.Array "jax.Array"))

- **repeats** (*ArrayLike*)

- **axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **total_repeat_length** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *PartitionSpec* *\|* *None*)

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.Array.real.html "previous page")

previous

jax.Array.real

[](jax.Array.reshape.html "next page")

next

jax.Array.reshape

Contents

- [`Array.repeat()`](#jax.Array.repeat)

By The JAX authors

© Copyright 2024, The JAX Authors.\
