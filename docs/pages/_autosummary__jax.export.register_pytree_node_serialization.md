- [](../index.html)
- [API Reference](../jax.html)
- [`jax.export` module](../jax.export.html)
- jax.export.register_pytree_node_serialization

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.export.register_pytree_node_serialization.rst "Download source file")
-  .pdf

# jax.export.register_pytree_node_serialization

## Contents

- [`register_pytree_node_serialization()`](#jax.export.register_pytree_node_serialization)

# jax.export.register_pytree_node_serialization[\#](#jax-export-register-pytree-node-serialization "Link to this heading")

jax.export.register_pytree_node_serialization(*nodetype*, *\**, *serialized_name*, *serialize_auxdata*, *deserialize_auxdata*, *from_children=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L404-L472)[\#](#jax.export.register_pytree_node_serialization "Link to this definition")  
Registers a custom PyTree node for serialization and deserialization.

You must use this function before you can serialize and deserialize PyTree nodes for the types not supported natively. We serialize PyTree nodes for the `in_tree` and `out_tree` fields of `Exported`, which are part of the exported function’s calling convention.

This function must be called after calling [`jax.tree_util.register_pytree_node()`](jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node") (except for `collections.namedtuple`, which do not require a call to `register_pytree_node`).

Parameters:  
- **nodetype** ([*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[T\]*) – the type whose PyTree nodes we want to serialize. It is an error to attempt to register multiple serializations for a `nodetype`.

- **serialized_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – a string that will be present in the serialization and will be used to look up the registration during deserialization. It is an error to attempt to register multiple serializations for a `serialized_name`.

- **serialize_auxdata** (*\_SerializeAuxData*) – serialize the PyTree auxdata (returned by the `flatten_func` argument to [`jax.tree_util.register_pytree_node()`](jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node").).

- **deserialize_auxdata** (*\_DeserializeAuxData*) – deserialize the auxdata that was serialized by the `serialize_auxdata`.

- **from_children** (*\_BuildFromChildren* *\|* *None*) – if present, this is a function that takes that result of `deserialize_auxdata` along with some children and creates an instance of `nodetype`. This is similar to the `unflatten_func` passed to [`jax.tree_util.register_pytree_node()`](jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node"). If not present, we look up and use the `unflatten_func`. This is needed for `collections.namedtuple`, which does not have a `register_pytree_node`, but it can be useful to override that function. Note that the result of `from_children` is only used with [`jax.tree_util.tree_structure()`](jax.tree_util.tree_structure.html#jax.tree_util.tree_structure "jax.tree_util.tree_structure") to construct a proper PyTree node, it is not used to construct the outputs of the serialized function.

Returns:  
the same type passed as `nodetype`, so that this function can be used as a class decorator.

Return type:  
[type](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")\[T\]

[](jax.export.default_export_platform.html "previous page")

previous

jax.export.default_export_platform

[](jax.export.register_namedtuple_serialization.html "next page")

next

jax.export.register_namedtuple_serialization

Contents

- [`register_pytree_node_serialization()`](#jax.export.register_pytree_node_serialization)

By The JAX authors

© Copyright 2024, The JAX Authors.\
