- [](../index.html)
- [API Reference](../jax.html)
- [`jax.export` module](../jax.export.html)
- jax.export.symbolic_shape

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.export.symbolic_shape.rst "Download source file")
-  .pdf

# jax.export.symbolic_shape

## Contents

- [`symbolic_shape()`](#jax.export.symbolic_shape)

# jax.export.symbolic_shape[\#](#jax-export-symbolic-shape "Link to this heading")

jax.export.symbolic_shape(*shape_spec*, *\**, *constraints=()*, *scope=None*, *like=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/shape_poly.py#L1381-L1433)[\#](#jax.export.symbolic_shape "Link to this definition")  
Constructs a symbolic shape from a string representation.

See [https://docs.jax.dev/en/latest/export/shape_poly.html](https://docs.jax.dev/en/latest/export/shape_poly.html) for examples.

Parameters:  
- **shape_spec** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *None*) – a symbolic shape specification. None stands for “…”. A shape specification is the string representation of a tuple (the parentheses are optional) with comma-separated dimension expressions. A dimension expression can be either: an integer constant, a dimension variable (alphanumeric starting with a letter), e1 + e2, e1 - e2, e1 \* e2, floordiv(e1, e2), mod(e1, e2), max(e1, e2), or min(e1, e2).

- **constraints** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) – a sequence of constraints on symbolic dimension expressions, of the form e1 \>= e2 or e1 \<= e2, or e1 == e2. See \[the documentation\]([https://docs.jax.dev/en/latest/export/shape_poly.html#user-specified-symbolic-constraints](https://docs.jax.dev/en/latest/export/shape_poly.html#user-specified-symbolic-constraints)) for usage.

- **scope** ([*SymbolicScope*](jax.export.SymbolicScope.html#jax.export.SymbolicScope "jax.export.SymbolicScope") *\|* *None*) – optionally, you can specify that the parsed symbolic expressions be created in the given scope. If this is missing, then a new SymbolicScope is created with the given constraints. You cannot specify both a scope and constraints (cannot add new constraints to a scope). See \[the documentation\]([https://docs.jax.dev/en/latest/export/shape_poly.html#user-specified-symbolic-constraints](https://docs.jax.dev/en/latest/export/shape_poly.html#user-specified-symbolic-constraints)) for usage.

- **like** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None\]* *\|* *None*) – when shape_spec contains placeholders (“\_”, “…”), use this shape to fill in the placeholders. The dimensions of like that are used for filling must be not None. If a dimension in like is not None and the corresponding dimension in shape_spec is a constant then they must be equal.

Return type:  
Sequence\[DimSize\]

Returns: a tuple with integers or symbolic expressions involving dimension variables.

[](jax.export.register_namedtuple_serialization.html "previous page")

previous

jax.export.register_namedtuple_serialization

[](jax.export.symbolic_args_specs.html "next page")

next

jax.export.symbolic_args_specs

Contents

- [`symbolic_shape()`](#jax.export.symbolic_shape)

By The JAX authors

© Copyright 2024, The JAX Authors.\
