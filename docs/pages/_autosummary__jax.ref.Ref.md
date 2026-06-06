- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.Ref

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.Ref.rst "Download source file")
-  .pdf

# jax.ref.Ref

## Contents

- [`Ref`](#jax.ref.Ref)
  - [`Ref.__init__()`](#jax.ref.Ref.__init__)

# jax.ref.Ref[\#](#jax-ref-ref "Link to this heading")

*class* jax.ref.Ref(*aval*, *refs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L2732-L2776)[\#](#jax.ref.Ref "Link to this definition")  
Mutable array reference.

In most cases this should not be constructed directly, but rather via [`jax.ref.new_ref()`](jax.ref.new_ref.html#jax.ref.new_ref "jax.ref.new_ref"). For examples of how this can be used, refer to the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html).

\_\_init\_\_(*aval*, *refs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/core.py#L2744-L2749)[\#](#jax.ref.Ref.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.ref.Ref.__init__ "jax.ref.Ref.__init__")(aval, refs) |  |
| `addupdate`(x\[, idx\]) |  |
| `unsafe_buffer_pointer`() |  |

Attributes

|             |     |
|-------------|-----|
| `at`        |     |
| `aval`      |     |
| `committed` |     |
| `dtype`     |     |
| `format`    |     |
| `ndim`      |     |
| `shape`     |     |
| `sharding`  |     |
| `size`      |     |

[](jax.ref.AbstractRef.html "previous page")

previous

jax.ref.AbstractRef

[](jax.ref.freeze.html "next page")

next

jax.ref.freeze

Contents

- [`Ref`](#jax.ref.Ref)
  - [`Ref.__init__()`](#jax.ref.Ref.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
