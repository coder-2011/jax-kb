- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree` module](../jax.tree.html)
- jax.tree.static

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree.static.rst "Download source file")
-  .pdf

# jax.tree.static

## Contents

- [`static()`](#jax.tree.static)

# jax.tree.static[\#](#jax-tree-static "Link to this heading")

jax.tree.static(*\*\*kwargs*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree.py#L469-L507)[\#](#jax.tree.static "Link to this definition")  
Convenience wrapper to declare a static pytree attribute.

Arguments are the same as those of [`dataclasses.field()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.field "(in Python v3.14)"), but [`static()`](#jax.tree.static "jax.tree.static") will automatically populate metadata with static = True, as used by [`jax.tree_util.register_dataclass()`](jax.tree_util.register_dataclass.html#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass").

Example

    >>> import jax
    >>> from dataclasses import dataclass
    ...
    >>> @jax.tree_util.register_dataclass
    ... @dataclass
    ... class MyOp:
    ...   x: jax.Array
    ...   y: jax.Array
    ...   op: str = jax.tree.static(default="add")  # static string field
    ...
    >>> m = MyOp(x=jnp.ones(3), y=jnp.arange(3))
    >>> m
    MyOp(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

    >>> leaves, treedef = jax.tree.flatten(m)
    >>> leaves
    [Array([1., 1., 1.], dtype=float32), Array([0, 1, 2], dtype=int32)]

    >>> treedef
    PyTreeDef(CustomNode(MyOp[('add',)], [*, *]))

    >>> jax.tree.unflatten(treedef, leaves)
    MyOp(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

See also

- [`jax.tree_util.register_dataclass()`](jax.tree_util.register_dataclass.html#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass")

[](jax.tree.reduce_associative.html "previous page")

previous

jax.tree.reduce_associative

[](jax.tree.structure.html "next page")

next

jax.tree.structure

Contents

- [`static()`](#jax.tree.static)

By The JAX authors

© Copyright 2024, The JAX Authors.\
