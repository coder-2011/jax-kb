- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.program_id

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.program_id.rst "Download source file")
-  .pdf

# jax.experimental.pallas.program_id

## Contents

- [`program_id()`](#jax.experimental.pallas.program_id)

# jax.experimental.pallas.program_id[\#](#jax-experimental-pallas-program-id "Link to this heading")

jax.experimental.pallas.program_id(*axis*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L62-L75)[\#](#jax.experimental.pallas.program_id "Link to this definition")  
Returns the kernel execution position along the given axis of the grid.

For example, with a 2D `grid` in the kernel execution corresponding to the grid coordinates `(1,`` ``2)`, `program_id(axis=0)` returns `1` and `program_id(axis=1)` returns `2`.

The returned value is an array of shape `()` and dtype `int32`.

Parameters:  
**axis** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – the axis of the grid along which to count the program.

Return type:  
jax_typing.Array

[](jax.experimental.pallas.pallas_call.html "previous page")

previous

jax.experimental.pallas.pallas_call

[](jax.experimental.pallas.num_programs.html "next page")

next

jax.experimental.pallas.num_programs

Contents

- [`program_id()`](#jax.experimental.pallas.program_id)

By The JAX authors

© Copyright 2024, The JAX Authors.\
