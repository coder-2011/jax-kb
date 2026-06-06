- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.sparse` module](../jax.experimental.sparse.html)
- jax.experimental.sparse.JAXSparse

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.sparse.JAXSparse.rst "Download source file")
-  .pdf

# jax.experimental.sparse.JAXSparse

## Contents

- [`JAXSparse`](#jax.experimental.sparse.JAXSparse)
  - [`JAXSparse.__init__()`](#jax.experimental.sparse.JAXSparse.__init__)

# jax.experimental.sparse.JAXSparse[\#](#jax-experimental-sparse-jaxsparse "Link to this heading")

*class* jax.experimental.sparse.JAXSparse(*args*, *\**, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/_base.py#L28-L134)[\#](#jax.experimental.sparse.JAXSparse "Link to this definition")  
Base class for high-level JAX sparse objects.

Parameters:  
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* *...\]*)

- **shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*)

\_\_init\_\_(*args*, *\**, *shape*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/_base.py#L54-L56)[\#](#jax.experimental.sparse.JAXSparse.__init__ "Link to this definition")  
Parameters:  
- **args** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* *...\]*)

- **shape** ([*Sequence*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*)

Methods

|  |  |
|----|----|
| [`__init__`](#jax.experimental.sparse.JAXSparse.__init__ "jax.experimental.sparse.JAXSparse.__init__")(args, \*, shape) |  |
| `block_until_ready`() |  |
| `sum`(\*args, \*\*kwargs) |  |
| `todense`() |  |
| `transpose`(\[axes\]) |  |
| `tree_flatten`() |  |
| `tree_unflatten`(aux_data, children) |  |

Attributes

|         |     |
|---------|-----|
| `T`     |     |
| `dtype` |     |
| `ndim`  |     |
| `nse`   |     |
| `size`  |     |
| `data`  |     |
| `shape` |     |

[](jax.experimental.sparse.random_bcoo.html "previous page")

previous

jax.experimental.sparse.random_bcoo

[](jax.experimental.sparse.BCOO.html "next page")

next

jax.experimental.sparse.BCOO

Contents

- [`JAXSparse`](#jax.experimental.sparse.JAXSparse)
  - [`JAXSparse.__init__()`](#jax.experimental.sparse.JAXSparse.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
