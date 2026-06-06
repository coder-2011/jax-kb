- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.register_pytree_node

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.register_pytree_node.rst "Download source file")
-  .pdf

# jax.tree_util.register_pytree_node

## Contents

- [`register_pytree_node()`](#jax.tree_util.register_pytree_node)

# jax.tree_util.register_pytree_node[\#](#jax-tree-util-register-pytree-node "Link to this heading")

jax.tree_util.register_pytree_node(*nodetype*, *flatten_func*, *unflatten_func*, *flatten_with_keys_func=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L255-L333)[\#](#jax.tree_util.register_pytree_node "Link to this definition")  
Extends the set of types that are considered internal nodes in pytrees.

See [example usage](../pytrees.html#pytrees).

Parameters:  
- **nodetype** ([*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[T\]*) – a Python type to register as a pytree.

- **flatten_func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T\],* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[\_Children,* *\_AuxData\]\]*) – a function to be used during flattening, taking a value of type `nodetype` and returning a pair, with (1) an iterable for the children to be flattened recursively, and (2) some hashable auxiliary data to be stored in the treedef and to be passed to the `unflatten_func`.

- **unflatten_func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[\_AuxData,* *\_Children\],* *T\]*) – a function taking two arguments: the auxiliary data that was returned by `flatten_func` and stored in the treedef, and the unflattened children. The function should return an instance of `nodetype`.

- **flatten_with_keys_func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[\[T\],* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[KeyLeafPairs,* *\_AuxData\]\]* *\|* *None*)

Return type:  
None

See also

- [`register_static()`](jax.tree_util.register_static.html#jax.tree_util.register_static "jax.tree_util.register_static"): simpler API for registering a static pytree.

- [`register_dataclass()`](jax.tree_util.register_dataclass.html#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass"): simpler API for registering a dataclass.

- [`register_pytree_with_keys()`](jax.tree_util.register_pytree_with_keys.html#jax.tree_util.register_pytree_with_keys "jax.tree_util.register_pytree_with_keys")

- [`register_pytree_node_class()`](jax.tree_util.register_pytree_node_class.html#jax.tree_util.register_pytree_node_class "jax.tree_util.register_pytree_node_class")

- [`register_pytree_with_keys_class()`](jax.tree_util.register_pytree_with_keys_class.html#jax.tree_util.register_pytree_with_keys_class "jax.tree_util.register_pytree_with_keys_class")

Examples

First we’ll define a custom type:

    >>> class MyContainer:
    ...   def __init__(self, size):
    ...     self.x = jnp.zeros(size)
    ...     self.y = jnp.ones(size)
    ...     self.size = size

If we try using this in a JIT-compiled function, we’ll get an error because JAX does not yet know how to handle this type:

    >>> m = MyContainer(size=5)
    >>> def f(m):
    ...   return m.x + m.y + jnp.arange(m.size)
    >>> jax.jit(f)(m)  
    Traceback (most recent call last):
      ...
    TypeError: Cannot interpret value of type <class 'jax.tree_util.MyContainer'> as an abstract array; it does not have a dtype attribute

In order to make our object recognized by JAX, we must register it as a pytree:

    >>> def flatten_func(obj):
    ...   children = (obj.x, obj.y)  # children must contain arrays & pytrees
    ...   aux_data = (obj.size,)  # aux_data must contain static, hashable data.
    ...   return (children, aux_data)
    ...
    >>> def unflatten_func(aux_data, children):
    ...   # Here we avoid `__init__` because it has extra logic we don't require:
    ...   obj = object.__new__(MyContainer)
    ...   obj.x, obj.y = children
    ...   obj.size, = aux_data
    ...   return obj
    ...
    >>> jax.tree_util.register_pytree_node(MyContainer, flatten_func, unflatten_func)

Now with this defined, we can use instances of this type in JIT-compiled functions.

    >>> jax.jit(f)(m)
    Array([1., 2., 3., 4., 5.], dtype=float32)

[](jax.tree_util.register_dataclass.html "previous page")

previous

jax.tree_util.register_dataclass

[](jax.tree_util.register_pytree_node_class.html "next page")

next

jax.tree_util.register_pytree_node_class

Contents

- [`register_pytree_node()`](#jax.tree_util.register_pytree_node)

By The JAX authors

© Copyright 2024, The JAX Authors.\
