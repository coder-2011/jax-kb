- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.debug_print

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.debug_print.rst "Download source file")
-  .pdf

# jax.experimental.pallas.debug_print

## Contents

- [`debug_print()`](#jax.experimental.pallas.debug_print)

# jax.experimental.pallas.debug_print[\#](#jax-experimental-pallas-debug-print "Link to this heading")

jax.experimental.pallas.debug_print(*fmt*, *\*args*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/primitives.py#L595-L617)[\#](#jax.experimental.pallas.debug_print "Link to this definition")  
Prints values from inside a Pallas kernel.

Parameters:  
- **fmt** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) –

  A format string to be included in the output. The restrictions on the format string depend on the backend:

  - On GPU, when using Triton, `fmt` must not contain any placeholders (`{...}`), since it is always printed before any of the values.

  - On GPU, when using the experimental Mosaic GPU backend, `fmt` must contain a placeholder for each value to be printed. Format specs and conversions are not supported. If a single value is provided, the value may be an array. Otherwise, all values must be scalars.

  - On TPU, if all inputs are scalars: If `fmt` contains placeholders, all values must be 32-bit integers. If there are no placeholders, the values are printed after the format string.

  - On TPU, if the input is a single vector, the vector is printed after the format string. The format string must end with a single placeholder `{}`.

- **\*args** (*jax_typing.ArrayLike*) – The values to print.

[](jax.experimental.pallas.debug_check.html "previous page")

previous

jax.experimental.pallas.debug_check

[](jax.experimental.pallas.dot.html "next page")

next

jax.experimental.pallas.dot

Contents

- [`debug_print()`](#jax.experimental.pallas.debug_print)

By The JAX authors

© Copyright 2024, The JAX Authors.\
