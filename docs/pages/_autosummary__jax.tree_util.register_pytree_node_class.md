- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.register_pytree_node_class

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.register_pytree_node_class.rst "Download source file")
-  .pdf

# jax.tree_util.register_pytree_node_class

## Contents

- [`register_pytree_node_class()`](#jax.tree_util.register_pytree_node_class)

# jax.tree_util.register_pytree_node_class[\#](#jax-tree-util-register-pytree-node-class "Link to this heading")

jax.tree_util.register_pytree_node_class(*cls*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L335-L385)[\#](#jax.tree_util.register_pytree_node_class "Link to this definition")  
Extends the set of types that are considered internal nodes in pytrees.

This function is a thin wrapper around `register_pytree_node`, and provides a class-oriented interface.

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

- [`register_pytree_with_keys_class()`](jax.tree_util.register_pytree_with_keys_class.html#jax.tree_util.register_pytree_with_keys_class "jax.tree_util.register_pytree_with_keys_class")

Examples

Here we’ll define a custom container that will be compatible with [`jax.jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations:

    >>> import jax
    >>> @jax.tree_util.register_pytree_node_class
    ... class MyContainer:
    ...   def __init__(self, x, y):
    ...     self.x = x
    ...     self.y = y
    ...   def tree_flatten(self):
    ...     return ((self.x, self.y), None)
    ...   @classmethod
    ...   def tree_unflatten(cls, aux_data, children):
    ...     return cls(*children)
    ...
    >>> m = MyContainer(jnp.zeros(4), jnp.arange(4))
    >>> def f(m):
    ...   return m.x + 2 * m.y
    >>> jax.jit(f)(m)
    Array([0., 2., 4., 6.], dtype=float32)

[](jax.tree_util.register_pytree_node.html "previous page")

previous

jax.tree_util.register_pytree_node

[](jax.tree_util.register_pytree_with_keys.html "next page")

next

jax.tree_util.register_pytree_with_keys

Contents

- [`register_pytree_node_class()`](#jax.tree_util.register_pytree_node_class)

By The JAX authors

© Copyright 2024, The JAX Authors.\
