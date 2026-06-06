- [](../index.html)
- [API Reference](../jax.html)
- jax.set_mesh

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.set_mesh.rst "Download source file")
-  .pdf

# jax.set_mesh

## Contents

- [`set_mesh`](#jax.set_mesh)
  - [`set_mesh.__init__()`](#jax.set_mesh.__init__)

# jax.set_mesh[\#](#jax-set-mesh "Link to this heading")

*class* jax.set_mesh(*mesh*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L977-L1024)[\#](#jax.set_mesh "Link to this definition")  
Sets a concrete mesh in a thread-local context.

`jax.set_mesh` has dual behavior. You can use it as a global setter or as a context manager.

When a mesh is in context via `jax.set_mesh`, you can use pass raw PartitionSpecs to all APIs that accept sharding as an argument. Using `jax.set_mesh` is also required for enabling explicit sharding mode: [https://docs.jax.dev/en/latest/parallel.html](https://docs.jax.dev/en/latest/parallel.html)

For example:

    mesh = jax.make_mesh((2,), ('x',))
    jax.set_mesh(mesh)  # use the API as a global setter

    with jax.set_mesh(mesh):  # use the API as a context manager
      ...

Note: `jax.set_mesh` can only be used outside of `jax.jit`.

Parameters:  
**mesh** ([*Mesh*](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") *\|* *None*)

\_\_init\_\_(*mesh*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/sharding_impls.py#L1000-L1017)[\#](#jax.set_mesh.__init__ "Link to this definition")  
Parameters:  
**mesh** ([*Mesh*](../jax.sharding.html#jax.sharding.Mesh "jax.sharding.Mesh") *\|* *None*)

Methods

|                                                                    |     |
|--------------------------------------------------------------------|-----|
| [`__init__`](#jax.set_mesh.__init__ "jax.set_mesh.__init__")(mesh) |     |

Attributes

|                      |     |
|----------------------|-----|
| `prev_abstract_mesh` |     |
| `prev_mesh`          |     |

[](jax.make_mesh.html "previous page")

previous

jax.make_mesh

[](jax.grad.html "next page")

next

jax.grad

Contents

- [`set_mesh`](#jax.set_mesh)
  - [`set_mesh.__init__()`](#jax.set_mesh.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
