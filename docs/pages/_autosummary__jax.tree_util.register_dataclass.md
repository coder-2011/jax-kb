- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.register_dataclass

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.register_dataclass.rst "Download source file")
-  .pdf

# jax.tree_util.register_dataclass

## Contents

- [`register_dataclass()`](#jax.tree_util.register_dataclass)

# jax.tree_util.register_dataclass[\#](#jax-tree-util-register-dataclass "Link to this heading")

jax.tree_util.register_dataclass(*nodetype*, *data_fields=None*, *meta_fields=None*, *drop_fields=()*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L999-L1171)[\#](#jax.tree_util.register_dataclass "Link to this definition")  
Extends the set of types that are considered internal nodes in pytrees.

This differs from `register_pytree_with_keys_class` in that the C++ registries use the optimized C++ dataclass builtin instead of the argument functions.

See [Custom pytree nodes](../custom_pytrees.html#pytrees-custom-pytree-nodes) for more information about registering pytrees.

Parameters:  
- **nodetype** (*Typ*) – a Python type to treat as an internal pytree node. This is assumed to have the semantics of a [`dataclass`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass "(in Python v3.14)"): namely, class attributes represent the whole of the object state, and can be passed as keywords to the class constructor to create a copy of the object. All defined attributes should be listed among `meta_fields` or `data_fields`.

- **meta_fields** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*) – metadata field names: these are attributes which will be treated as [static](../glossary.html#term-static) when this pytree is passed to [`jax.jit()`](jax.jit.html#jax.jit "jax.jit"). `meta_fields` is optional only if `nodetype` is a dataclass, in which case individual fields can be marked static via [`dataclasses.field()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.field "(in Python v3.14)") (see examples below). Metadata fields *must* be static, hashable, immutable objects, as these objects are used to generate JIT cache keys. In particular, metadata fields cannot contain [`jax.Array`](jax.Array.html#jax.Array "jax.Array") or [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") objects.

- **data_fields** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]* *\|* *None*) – data field names: these are attributes which will be treated as non-static when this pytree is passed to [`jax.jit()`](jax.jit.html#jax.jit "jax.jit"). `data_fields` is optional only if `nodetype` is a dataclass, in which case fields are assumed data fields unless marked via [`dataclasses.field()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.field "(in Python v3.14)") (see examples below). Data fields *must* be JAX-compatible objects such as arrays ([`jax.Array`](jax.Array.html#jax.Array "jax.Array") or [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)")), scalars, or pytrees whose leaves are arrays or scalars. Note that `None` is a valid data field, as JAX recognizes this as an empty pytree.

- **drop_fields** (*Sequence\[*[*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")*\]*) – only referenced if `nodetype` is a dataclass. Specify a sequence of field names from among `dataclasses.fields(nodetype)` to be excluded from pytree registration.

Returns:  
The input class `nodetype` is returned unchanged after being added to JAX’s pytree registry, so that [`register_dataclass()`](#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass") can be used as a decorator.

Return type:  
Typ

Examples

In JAX v0.4.35 or older, you must specify `data_fields` and `meta_fields` in order to use this decorator:

    >>> import jax
    >>> from dataclasses import dataclass
    >>> from functools import partial
    ...
    >>> @partial(jax.tree_util.register_dataclass,
    ...          data_fields=['x', 'y'],
    ...          meta_fields=['op'])
    ... @dataclass
    ... class MyStruct:
    ...   x: jax.Array
    ...   y: jax.Array
    ...   op: str
    ...
    >>> m = MyStruct(x=jnp.ones(3), y=jnp.arange(3), op='add')
    >>> m
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

Starting in JAX v0.4.36, the `data_fields` and `meta_fields` arguments are optional for [`dataclass()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.dataclass "(in Python v3.14)") inputs, with fields defaulting to `data_fields` unless marked as static using static metadata in [`dataclasses.field()`](https://docs.python.org/3/library/dataclasses.html#dataclasses.field "(in Python v3.14)").

    >>> import jax
    >>> from dataclasses import dataclass, field
    ...
    >>> @jax.tree_util.register_dataclass
    ... @dataclass
    ... class MyStruct:
    ...   x: jax.Array  # defaults to non-static data field
    ...   y: jax.Array  # defaults to non-static data field
    ...   op: str = field(metadata=dict(static=True))  # marked as static meta field.
    ...
    >>> m = MyStruct(x=jnp.ones(3), y=jnp.arange(3), op='add')
    >>> m
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

Once this class is registered, it can be used with functions in [`jax.tree`](../jax.tree.html#module-jax.tree "jax.tree") and [`jax.tree_util`](../jax.tree_util.html#module-jax.tree_util "jax.tree_util"):

    >>> leaves, treedef = jax.tree.flatten(m)
    >>> leaves
    [Array([1., 1., 1.], dtype=float32), Array([0, 1, 2], dtype=int32)]
    >>> treedef
    PyTreeDef(CustomNode(MyStruct[('add',)], [*, *]))
    >>> jax.tree.unflatten(treedef, leaves)
    MyStruct(x=Array([1., 1., 1.], dtype=float32), y=Array([0, 1, 2], dtype=int32), op='add')

In particular, this registration allows `m` to be passed seamlessly through code wrapped in [`jax.jit()`](jax.jit.html#jax.jit "jax.jit") and other JAX transformations, with `data_fields` being treated as dynamic arguments, and `meta_fields` being treated as static arguments:

    >>> @jax.jit
    ... def compiled_func(m):
    ...   if m.op == 'add':
    ...     return m.x + m.y
    ...   else:
    ...     raise ValueError(f"{m.op=}")
    ...
    >>> compiled_func(m)
    Array([1., 2., 3.], dtype=float32)

[](jax.tree_util.is_tree_node.html "previous page")

previous

jax.tree_util.is_tree_node

[](jax.tree_util.register_pytree_node.html "next page")

next

jax.tree_util.register_pytree_node

Contents

- [`register_dataclass()`](#jax.tree_util.register_dataclass)

By The JAX authors

© Copyright 2024, The JAX Authors.\
