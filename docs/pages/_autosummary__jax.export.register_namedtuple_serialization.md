- [](../index.html)
- [API Reference](../jax.html)
- [`jax.export` module](../jax.export.html)
- jax.export.register_namedtuple_serialization

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.export.register_namedtuple_serialization.rst "Download source file")
-  .pdf

# jax.export.register_namedtuple_serialization

## Contents

- [`register_namedtuple_serialization()`](#jax.export.register_namedtuple_serialization)

# jax.export.register_namedtuple_serialization[\#](#jax-export-register-namedtuple-serialization "Link to this heading")

jax.export.register_namedtuple_serialization(*nodetype*, *\**, *serialized_name*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/export/_export.py#L474-L525)[\#](#jax.export.register_namedtuple_serialization "Link to this definition")  
Registers a namedtuple for serialization and deserialization.

JAX has native PyTree support for `collections.namedtuple`, and does not require a call to [`jax.tree_util.register_pytree_node()`](jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node"). However, if you want to serialize functions that have inputs of outputs of a namedtuple type, you must register that type for serialization.

Parameters:  
- **nodetype** ([*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[T\]*) – the type whose PyTree nodes we want to serialize. It is an error to attempt to register multiple serializations for a `nodetype`. On deserialization, this type must have the same set of keys that were present during serialization.

- **serialized_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – a string that will be present in the serialization and will be used to look up the registration during deserialization. It is an error to attempt to register multiple serializations for a `serialized_name`.

Returns:  
the same type passed as `nodetype`, so that this function can be used as a class decorator.

Return type:  
[type](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")\[T\]

[](jax.export.register_pytree_node_serialization.html "previous page")

previous

jax.export.register_pytree_node_serialization

[](jax.export.symbolic_shape.html "next page")

next

jax.export.symbolic_shape

Contents

- [`register_namedtuple_serialization()`](#jax.export.register_namedtuple_serialization)

By The JAX authors

© Copyright 2024, The JAX Authors.\
