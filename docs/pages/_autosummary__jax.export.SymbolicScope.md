- [](../index.html)
- [API Reference](../jax.html)
- [`jax.export` module](../jax.export.html)
- jax.export.SymbolicScope

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.export.SymbolicScope.rst "Download source file")
-  .pdf

# jax.export.SymbolicScope

## Contents

- [`SymbolicScope`](#jax.export.SymbolicScope)
  - [`SymbolicScope.__init__()`](#jax.export.SymbolicScope.__init__)

# jax.export.SymbolicScope[\#](#jax-export-symbolicscope "Link to this heading")

*class* jax.export.SymbolicScope(*constraints_str=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/shape_poly.py#L980-L1120)[\#](#jax.export.SymbolicScope "Link to this definition")  
Identifies a scope for symbolic expressions.

All symbolic expressions that interact (e.g., appear in the argument shapes for one JAX function invocation, or are involved in arithmetic operations) must be from the same scope and must share the same SymbolicScope object.

Holds the constraints on symbolic expressions.

See \[the README\]([https://docs.jax.dev/en/latest/export/shape_poly.html#user-specified-symbolic-constraints](https://docs.jax.dev/en/latest/export/shape_poly.html#user-specified-symbolic-constraints)) for more details.

Parameters:  
**constraints_str** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) – A sequence of constraints on symbolic dimension expressions, of the form e1 \>= e2 or e1 \<= e2 or e1 == e2.

\_\_init\_\_(*constraints_str=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/shape_poly.py#L997-L1031)[\#](#jax.export.SymbolicScope.__init__ "Link to this definition")  
Parameters:  
**constraints_str** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.export.SymbolicScope.__init__ "jax.export.SymbolicScope.__init__")(\[constraints_str\]) |  |

[](jax.export.is_symbolic_dim.html "previous page")

previous

jax.export.is_symbolic_dim

[](../jax.extend.html "next page")

next

`jax.extend` module

Contents

- [`SymbolicScope`](#jax.export.SymbolicScope)
  - [`SymbolicScope.__init__()`](#jax.export.SymbolicScope.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
