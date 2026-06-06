- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.cdiv

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.cdiv.rst "Download source file")
-  .pdf

# jax.experimental.pallas.cdiv

## Contents

- [`cdiv()`](#jax.experimental.pallas.cdiv)

# jax.experimental.pallas.cdiv[\#](#jax-experimental-pallas-cdiv "Link to this heading")

jax.experimental.pallas.cdiv(*a: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*, *b: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*) → [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/utils.py#L47-L59)[\#](#jax.experimental.pallas.cdiv "Link to this definition")\
jax.experimental.pallas.cdiv(*a: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*, *b: jax_typing.Array*) → jax_typing.Array\
jax.experimental.pallas.cdiv(*a: jax_typing.Array*, *b: [int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*) → jax_typing.Array\
jax.experimental.pallas.cdiv(*a: jax_typing.Array*, *b: jax_typing.Array*) → jax_typing.Array  
Computes the ceiling division of a divided by b.

Examples

    >>> cdiv(8, 2)
    4
    >>> cdiv(9, 2)  # 9 / 2 = 4.5, which rounds up to 5
    5

[](jax.experimental.pallas.num_programs.html "previous page")

previous

jax.experimental.pallas.num_programs

[](jax.experimental.pallas.dslice.html "next page")

next

jax.experimental.pallas.dslice

Contents

- [`cdiv()`](#jax.experimental.pallas.cdiv)

By The JAX authors

© Copyright 2024, The JAX Authors.\
