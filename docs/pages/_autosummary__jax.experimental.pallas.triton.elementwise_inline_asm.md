- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- [`jax.experimental.pallas.triton` module](../jax.experimental.pallas.triton.html)
- jax.experimental.pallas.triton.elementwise_inline_asm

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.triton.elementwise_inline_asm.rst "Download source file")
-  .pdf

# jax.experimental.pallas.triton.elementwise_inline_asm

## Contents

- [`elementwise_inline_asm()`](#jax.experimental.pallas.triton.elementwise_inline_asm)

# jax.experimental.pallas.triton.elementwise_inline_asm[\#](#jax-experimental-pallas-triton-elementwise-inline-asm "Link to this heading")

jax.experimental.pallas.triton.elementwise_inline_asm(*asm*, *\**, *args*, *constraints*, *pack*, *result_shape_dtypes*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/triton/primitives.py#L82-L112)[\#](#jax.experimental.pallas.triton.elementwise_inline_asm "Link to this definition")  
Inline assembly applying an elementwise operation.

Parameters:  
- **asm** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The assembly code to run.

- **args** (*Sequence\[*[*jax.Array*](jax.Array.html#jax.Array "jax.Array")*\]*) – The arguments to pass to the assembly code.

- **constraints** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – LLVM inline assembly [constraints](https://llvm.org/docs/LangRef.html#inline-asm-constraint-string).

- **pack** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – The number of elements from each argument expected by a single instance of the assembly code.

- **result_shape_dtypes** (*Sequence\[*[*jax.ShapeDtypeStruct*](jax.ShapeDtypeStruct.html#jax.ShapeDtypeStruct "jax.ShapeDtypeStruct")*\]*) – The shapes and dtypes of the results produced by the assembly code.

Returns:  
The results produced by the assembly code.

Return type:  
Sequence\[[jax.Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.experimental.pallas.triton.debug_barrier.html "previous page")

previous

jax.experimental.pallas.triton.debug_barrier

[](jax.experimental.pallas.triton.load.html "next page")

next

jax.experimental.pallas.triton.load

Contents

- [`elementwise_inline_asm()`](#jax.experimental.pallas.triton.elementwise_inline_asm)

By The JAX authors

© Copyright 2024, The JAX Authors.\
