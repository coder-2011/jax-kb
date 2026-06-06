- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.isdtype

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.isdtype.rst "Download source file")
-  .pdf

# jax.numpy.isdtype

## Contents

- [`isdtype()`](#jax.numpy.isdtype)

# jax.numpy.isdtype[\#](#jax-numpy-isdtype "Link to this heading")

jax.numpy.isdtype(*dtype*, *kind*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/dtypes.py#L666-L710)[\#](#jax.numpy.isdtype "Link to this definition")  
Returns a boolean indicating whether a provided dtype is of a specified kind.

Parameters:  
- **dtype** (*DTypeLike*) – the input dtype

- **kind** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *DTypeLike* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)") *\|* *DTypeLike,* *...\]*) –

  the data type kind. If `kind` is dtype-like, return `dtype`` ``=`` ``kind`. If `kind` is a string, then return True if the dtype is in the specified category:

  - `'bool'`: `{bool}`

  - `'signed`` ``integer'`: `{int4,`` ``int8,`` ``int16,`` ``int32,`` ``int64}`

  - `'unsigned`` ``integer'`: `{uint4,`` ``uint8,`` ``uint16,`` ``uint32,`` ``uint64}`

  - `'integral'`: shorthand for `('signed`` ``integer',`` ``'unsigned`` ``integer')`

  - `'real`` ``floating'`: `{float8_*,`` ``float16,`` ``bfloat16,`` ``float32,`` ``float64}`

  - `'complex`` ``floating'`: `{complex64,`` ``complex128}`

  - `'numeric'`: shorthand for `('integral',`` ``'real`` ``floating',`` ``'complex`` ``floating')`

  If `kind` is a tuple, then return True if dtype matches any entry of the tuple.

Returns:  
True or False

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

[](jax.numpy.iscomplexobj.html "previous page")

previous

jax.numpy.iscomplexobj

[](jax.numpy.isfinite.html "next page")

next

jax.numpy.isfinite

Contents

- [`isdtype()`](#jax.numpy.isdtype)

By The JAX authors

© Copyright 2024, The JAX Authors.\
