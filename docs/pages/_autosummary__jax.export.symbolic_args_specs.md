- [](../index.html)
- [API Reference](../jax.html)
- [`jax.export` module](../jax.export.html)
- jax.export.symbolic_args_specs

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.export.symbolic_args_specs.rst "Download source file")
-  .pdf

# jax.export.symbolic_args_specs

## Contents

- [`symbolic_args_specs()`](#jax.export.symbolic_args_specs)

# jax.export.symbolic_args_specs[\#](#jax-export-symbolic-args-specs "Link to this heading")

jax.export.symbolic_args_specs(*args*, *shapes_specs*, *constraints=()*, *scope=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/shape_poly.py#L1434-L1495)[\#](#jax.export.symbolic_args_specs "Link to this definition")  
Constructs a pytree of jax.ShapeDtypeStruct arguments specs for export.

See the documentation of [`jax.export.symbolic_shape()`](jax.export.symbolic_shape.html#jax.export.symbolic_shape "jax.export.symbolic_shape") and the \[shape polymorphism documentation\]([https://docs.jax.dev/en/latest/export/shape_poly.html](https://docs.jax.dev/en/latest/export/shape_poly.html)) for details.

Parameters:  
- **args** – a pytree of arguments. These can be jax.Array, or jax.ShapeDtypeStruct. They are used to learn the pytree structure of the arguments, their dtypes, and to fill-in the actual shapes where the shapes_specs contains placeholders. Note that only the shape dimensions for which shapes_specs is a placeholder are used from args.

- **shapes_specs** – should be None (all arguments have static shapes), a single string (see shape_spec for [`jax.export.symbolic_shape()`](jax.export.symbolic_shape.html#jax.export.symbolic_shape "jax.export.symbolic_shape"); applies to all arguments), or a pytree matching a prefix of the args. See \[how optional parameters are matched to arguments\]([https://docs.jax.dev/en/latest/pytrees.html#applying-optional-parameters-to-pytrees](https://docs.jax.dev/en/latest/pytrees.html#applying-optional-parameters-to-pytrees)).

- **constraints** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) – as for [`jax.export.symbolic_shape()`](jax.export.symbolic_shape.html#jax.export.symbolic_shape "jax.export.symbolic_shape").

- **scope** ([*SymbolicScope*](jax.export.SymbolicScope.html#jax.export.SymbolicScope "jax.export.SymbolicScope") *\|* *None*) – as for [`jax.export.symbolic_shape()`](jax.export.symbolic_shape.html#jax.export.symbolic_shape "jax.export.symbolic_shape").

Returns: a pytree of jax.ShapeDtypeStruct matching the args with the shapes  
replaced with symbolic dimensions as specified by shapes_specs.

[](jax.export.symbolic_shape.html "previous page")

previous

jax.export.symbolic_shape

[](jax.export.is_symbolic_dim.html "next page")

next

jax.export.is_symbolic_dim

Contents

- [`symbolic_args_specs()`](#jax.export.symbolic_args_specs)

By The JAX authors

© Copyright 2024, The JAX Authors.\
