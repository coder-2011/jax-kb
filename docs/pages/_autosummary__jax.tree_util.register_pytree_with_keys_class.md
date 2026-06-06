- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.register_pytree_with_keys_class

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.register_pytree_with_keys_class.rst "Download source file")
-  .pdf

# jax.tree_util.register_pytree_with_keys_class

## Contents

- [`register_pytree_with_keys_class()`](#jax.tree_util.register_pytree_with_keys_class)

# jax.tree_util.register_pytree_with_keys_class[\#](#jax-tree-util-register-pytree-with-keys-class "Link to this heading")

jax.tree_util.register_pytree_with_keys_class(*cls*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L950-L997)[\#](#jax.tree_util.register_pytree_with_keys_class "Link to this definition")  
Extends the set of types that are considered internal nodes in pytrees.

This function is similar to `register_pytree_node_class`, but requires a class that defines how it could be flattened with keys.

It is a thin wrapper around `register_pytree_with_keys`, and provides a class-oriented interface:

Parameters:  
**cls** (*Typ*) – a type to register as a pytree

Returns:  
The input class `cls` is returned unchanged after being added to JAX’s pytree registry. This return value allows `register_pytree_node_class` to be used as a decorator.

Return type:  
Typ

See also

- [`register_static()`](jax.tree_util.register_static.html#jax.tree_util.register_static "jax.tree_util.register_static"): simpler API for registering a static pytree.

- [`register_dataclass()`](jax.tree_util.register_dataclass.html#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass"): simpler API for registering a dataclass.

- [`register_pytree_node()`](jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node")

- [`register_pytree_with_keys()`](jax.tree_util.register_pytree_with_keys.html#jax.tree_util.register_pytree_with_keys "jax.tree_util.register_pytree_with_keys")

- [`register_pytree_node_class()`](jax.tree_util.register_pytree_node_class.html#jax.tree_util.register_pytree_node_class "jax.tree_util.register_pytree_node_class")

Examples

    >>> from jax.tree_util import register_pytree_with_keys_class, GetAttrKey
    >>> @register_pytree_with_keys_class
    ... class Special:
    ...   def __init__(self, x, y):
    ...     self.x = x
    ...     self.y = y
    ...   def tree_flatten_with_keys(self):
    ...     return (((GetAttrKey('x'), self.x), (GetAttrKey('y'), self.y)), None)
    ...   @classmethod
    ...   def tree_unflatten(cls, aux_data, children):
    ...     return cls(*children)

[](jax.tree_util.register_pytree_with_keys.html "previous page")

previous

jax.tree_util.register_pytree_with_keys

[](jax.tree_util.register_static.html "next page")

next

jax.tree_util.register_static

Contents

- [`register_pytree_with_keys_class()`](#jax.tree_util.register_pytree_with_keys_class)

By The JAX authors

© Copyright 2024, The JAX Authors.\
