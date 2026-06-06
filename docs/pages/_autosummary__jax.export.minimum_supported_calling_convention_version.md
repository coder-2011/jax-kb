- [](../index.html)
- [API Reference](../jax.html)
- [`jax.export` module](../jax.export.html)
- jax.export.minimum_supported_calling_convention_version

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.export.minimum_supported_calling_convention_version.rst "Download source file")
-  .pdf

# jax.export.minimum_supported_calling_convention_version

## Contents

- [`minimum_supported_calling_convention_version`](#jax.export.minimum_supported_calling_convention_version)

# jax.export.minimum_supported_calling_convention_version[\#](#jax-export-minimum-supported-calling-convention-version "Link to this heading")

jax.export.minimum_supported_calling_convention_version *= 9*[\#](#jax.export.minimum_supported_calling_convention_version "Link to this definition")  
int(\[x\]) -\> integer int(x, base=10) -\> integer

Convert a number or string to an integer, or return 0 if no arguments are given. If x is a number, return x.\_\_int\_\_(). For floating-point numbers, this truncates towards zero.

If x is not a number or if base is given, then x must be a string, bytes, or bytearray instance representing an integer literal in the given base. The literal can be preceded by ‘+’ or ‘-’ and be surrounded by whitespace. The base defaults to 10. Valid bases are 0 and 2-36. Base 0 means to interpret the base from the string as an integer literal. \>\>\> int(‘0b100’, base=0) 4

[](jax.export.deserialize.html "previous page")

previous

jax.export.deserialize

[](jax.export.maximum_supported_calling_convention_version.html "next page")

next

jax.export.maximum_supported_calling_convention_version

Contents

- [`minimum_supported_calling_convention_version`](#jax.export.minimum_supported_calling_convention_version)

By The JAX authors

© Copyright 2024, The JAX Authors.\
