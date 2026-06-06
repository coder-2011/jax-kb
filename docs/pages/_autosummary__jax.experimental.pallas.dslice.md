- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.dslice

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.dslice.rst "Download source file")
-  .pdf

# jax.experimental.pallas.dslice

## Contents

- [`dslice()`](#jax.experimental.pallas.dslice)

# jax.experimental.pallas.dslice[\#](#jax-experimental-pallas-dslice "Link to this heading")

jax.experimental.pallas.dslice(*start*, *size=None*, *stride=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/indexing.py#L75-L116)[\#](#jax.experimental.pallas.dslice "Link to this definition")  
Constructs a `Slice` from a start index and a size.

The semantics of `dslice` mirror those of the builtin `slice` type:

- `dslice(None)` is `:`

- `dslice(j)` is `:j`

- `dslice(i,`` ``j)` is `i:i+j`

- `dslice(i,`` ``j,`` ``stride)` is `i:i+j:stride`

Examples

    >>> x = jax.numpy.arange(10)
    >>> i = 4
    >>> x[i: i + 2]  # standard indexing requires i to be static
    Array([4, 5], dtype=int32)
    >>> x[jax.ds(i, 2)]  # equivalent which allows i to be dynamic
    Array([4, 5], dtype=int32)

Here is an explicit example of slicing with a dynamic start index:

    >>> @jax.jit(static_argnames='size')
    ... def f(x, i, size):  # example of when `
    ...   return x[jax.ds(i, size)]
    ...
    >>> f(x, i, 2)
    Array([4, 5], dtype=int32)

Parameters:  
- **start** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *None*)

- **stride** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*)

Return type:  
[slice](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)") \| [Slice](jax.experimental.pallas.Slice.html#jax.experimental.pallas.Slice "jax.experimental.pallas.Slice")

[](jax.experimental.pallas.cdiv.html "previous page")

previous

jax.experimental.pallas.cdiv

[](jax.experimental.pallas.empty.html "next page")

next

jax.experimental.pallas.empty

Contents

- [`dslice()`](#jax.experimental.pallas.dslice)

By The JAX authors

© Copyright 2024, The JAX Authors.\
