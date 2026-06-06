- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.mlir` module](../jax.extend.mlir.html)
- jax.extend.mlir.refine_polymorphic_shapes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.mlir.refine_polymorphic_shapes.rst "Download source file")
-  .pdf

# jax.extend.mlir.refine_polymorphic_shapes

## Contents

- [`refine_polymorphic_shapes`](#jax.extend.mlir.refine_polymorphic_shapes)

# jax.extend.mlir.refine_polymorphic_shapes[\#](#jax-extend-mlir-refine-polymorphic-shapes "Link to this heading")

jax.extend.mlir.refine_polymorphic_shapes *= \<nanobind.nb_func object\>*[\#](#jax.extend.mlir.refine_polymorphic_shapes "Link to this definition")  
bool = True, validate_static_shapes: bool = True, enable_shardy: bool = False) -\> bytes

Refines the dynamic shapes for a module.  
The “main” function must have static shapes and all the intermediate dynamic shapes depend only on the input static shapes. Optionally, also validates that the resulting module has only static shapes.

Type:  
refine_polymorphic_shapes(mlir_module

Type:  
[bytes](https://docs.python.org/3/library/stdtypes.html#bytes "(in Python v3.14)"), enable_shape_assertions

[](jax.extend.mlir.passmanager.html "previous page")

previous

jax.extend.mlir.passmanager

[](jax.extend.mlir.serialize_portable_artifact.html "next page")

next

jax.extend.mlir.serialize_portable_artifact

Contents

- [`refine_polymorphic_shapes`](#jax.extend.mlir.refine_polymorphic_shapes)

By The JAX authors

© Copyright 2024, The JAX Authors.\
