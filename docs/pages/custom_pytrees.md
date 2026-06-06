- [](index.html)
- [Resources and Advanced Guides](advanced_guides.html)
- Custom pytree nodes

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/custom_pytrees.md "Download source file")
-  .pdf

# Custom pytree nodes

## Contents

- [Custom pytrees and initialization with unexpected values](#custom-pytrees-and-initialization-with-unexpected-values)
- [Internal pytree handling](#internal-pytree-handling)

# Custom pytree nodes[\#](#custom-pytree-nodes "Link to this heading")

This section explains how in JAX you can extend the set of Python types that will be considered *internal nodes* in pytrees (pytree nodes) by using [`jax.tree_util.register_pytree_node()`](_autosummary/jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node") with [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map").

Why would you need this? In the previous examples, pytrees were shown as lists, tuples, and dicts, with everything else as pytree leaves. This is because if you define your own container class, it will be considered to be a pytree leaf unless you *register* it with JAX. This is also the case even if your container class has trees inside it. For example:

    import jax

    class Special(object):
      def __init__(self, x, y):
        self.x = x
        self.y = y

    jax.tree.leaves([
        Special(0, 1),
        Special(2, 4),
    ])

    [<__main__.Special at 0x7075a95fab40>, <__main__.Special at 0x7075a946b020>]

Accordingly, if you try to use a [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") expecting the leaves to be elements inside the container, you will get an error:

    jax.tree.map(lambda x: x + 1,
      [
        Special(0, 1),
        Special(2, 4)
      ])

    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    Cell In[2], line 1
    ----> 1 jax.tree.map(lambda x: x + 1,
          2   [
          3     Special(0, 1),
          4     Special(2, 4)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/tree.py:156, in map(f, tree, is_leaf, *rest)
        116 def map(f: Callable[..., Any],
        117         tree: Any,
        118         *rest: Any,
        119         is_leaf: Callable[[Any], bool] | None = None) -> Any:
        120   """Maps a multi-input function over pytree args to produce a new pytree.
        121 
        122   Args:
       (...)    154     - :func:`jax.tree.reduce`
        155   """
    --> 156   return tree_util.tree_map(f, tree, *rest, is_leaf=is_leaf)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/tree_util.py:399, in tree_map(f, tree, is_leaf, *rest)
        397   err = next(_prefix_error((), tree, r2, is_leaf), None)  # type: ignore
        398   raise (err('tree_map tree') if err is not None else e) from None
    --> 399 return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/tree_util.py:399, in <genexpr>(.0)
        397   err = next(_prefix_error((), tree, r2, is_leaf), None)  # type: ignore
        398   raise (err('tree_map tree') if err is not None else e) from None
    --> 399 return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))

    Cell In[2], line 1, in <lambda>(x)
    ----> 1 jax.tree.map(lambda x: x + 1,
          2   [
          3     Special(0, 1),
          4     Special(2, 4)

    TypeError: unsupported operand type(s) for +: 'Special' and 'int'

As a solution, JAX allows to extend the set of types to be considered internal pytree nodes through a global registry of types. Additionally, the values of registered types are traversed recursively.

First, register a new type using [`jax.tree_util.register_pytree_node()`](_autosummary/jax.tree_util.register_pytree_node.html#jax.tree_util.register_pytree_node "jax.tree_util.register_pytree_node"):

    from jax.tree_util import register_pytree_node

    class RegisteredSpecial(Special):
      def __repr__(self):
        return "RegisteredSpecial(x={}, y={})".format(self.x, self.y)

    def special_flatten(v):
      """Specifies a flattening recipe.

      Params:
        v: The value of the registered type to flatten.
      Returns:
        A pair of an iterable with the children to be flattened recursively,
        and some opaque auxiliary data to pass back to the unflattening recipe.
        The auxiliary data is stored in the treedef for use during unflattening.
        The auxiliary data could be used, for example, for dictionary keys.
      """
      children = (v.x, v.y)
      aux_data = None
      return (children, aux_data)

    def special_unflatten(aux_data, children):
      """Specifies an unflattening recipe.

      Params:
        aux_data: The opaque data that was specified during flattening of the
          current tree definition.
        children: The unflattened children

      Returns:
        A reconstructed object of the registered type, using the specified
        children and auxiliary data.
      """
      return RegisteredSpecial(*children)

    # Global registration
    register_pytree_node(
        RegisteredSpecial,
        special_flatten,    # Instruct JAX what are the children nodes.
        special_unflatten   # Instruct JAX how to pack back into a `RegisteredSpecial`.
    )

Now you can traverse the special container structure:

    jax.tree.map(lambda x: x + 1,
      [
       RegisteredSpecial(0, 1),
       RegisteredSpecial(2, 4),
      ])

    [RegisteredSpecial(x=1, y=2), RegisteredSpecial(x=3, y=5)]

Alternatively, you can define appropriate `tree_flatten` and `tree_unflatten` methods on your class and decorate it with [`register_pytree_node_class()`](_autosummary/jax.tree_util.register_pytree_node_class.html#jax.tree_util.register_pytree_node_class "jax.tree_util.register_pytree_node_class"):

    from jax.tree_util import register_pytree_node_class

    @register_pytree_node_class
    class RegisteredSpecial2(Special):
      def __repr__(self):
        return "RegisteredSpecial2(x={}, y={})".format(self.x, self.y)

      def tree_flatten(self):
        children = (self.x, self.y)
        aux_data = None
        return (children, aux_data)

      @classmethod
      def tree_unflatten(cls, aux_data, children):
        return cls(*children)


    def show_example(structured):
      flat, tree = structured.tree_flatten()
      unflattened = RegisteredSpecial2.tree_unflatten(tree, flat)
      print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")


    show_example(RegisteredSpecial2(1., 2.))

    structured=RegisteredSpecial2(x=1.0, y=2.0)
      flat=(1.0, 2.0)
      tree=None
      unflattened=RegisteredSpecial2(x=1.0, y=2.0)

Modern Python comes equipped with helpful tools to make defining containers easier. Some will work with JAX out-of-the-box, but others require more care.

For instance, a Python `NamedTuple` subclass doesn’t need to be registered to be considered a pytree node type:

    from typing import NamedTuple, Any

    class MyOtherContainer(NamedTuple):
      name: str
      a: Any
      b: Any
      c: Any

    # NamedTuple subclasses are handled as pytree nodes, so
    # this will work out-of-the-box.
    jax.tree.leaves([
        MyOtherContainer('Alice', 1, 2, 3),
        MyOtherContainer('Bob', 4, 5, 6)
    ])

    ['Alice', 1, 2, 3, 'Bob', 4, 5, 6]

Notice that the `name` field now appears as a leaf, because all tuple elements are children. This is what happens when you don’t have to register the class the hard way.

When defining unflattening functions, in general `children` should contain all the dynamic elements of the data structure (arrays, dynamic scalars, and pytrees), while `aux_data` should contain all the static elements that will be rolled into the `treedef` structure. JAX sometimes needs to compare `treedef` for equality, or compute its hash for use in the JIT cache, and so care must be taken to ensure that the auxiliary data specified in the flattening recipe supports meaningful hashing and equality comparisons.

Unlike `NamedTuple` subclasses, classes decorated with `@dataclass` are not automatically pytrees. However, they can be registered as pytrees using the [`jax.tree_util.register_dataclass()`](_autosummary/jax.tree_util.register_dataclass.html#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass") decorator:

    from dataclasses import dataclass
    import jax.numpy as jnp
    import numpy as np
    import functools

    @functools.partial(jax.tree_util.register_dataclass,
                       data_fields=['a', 'b', 'c'],
                       meta_fields=['name'])
    @dataclass
    class MyDataclassContainer(object):
      name: str
      a: Any
      b: Any
      c: Any

    # MyDataclassContainer is now a pytree node.
    jax.tree.leaves([
      MyDataclassContainer('apple', 5.3, 1.2, jnp.zeros([4])),
      MyDataclassContainer('banana', np.array([3, 4]), -1., 0.)
    ])

    [5.3, 1.2, Array([0., 0., 0., 0.], dtype=float32), array([3, 4]), -1.0, 0.0]

Notice that the `name` field does not appear as a leaf. This is because we included it in the `meta_fields` argument to [`jax.tree_util.register_dataclass()`](_autosummary/jax.tree_util.register_dataclass.html#jax.tree_util.register_dataclass "jax.tree_util.register_dataclass"), indicating that it should be treated as metadata/auxiliary data, just like `aux_data` in `RegisteredSpecial` above. Now instances of `MyDataclassContainer` can be passed into JIT-ed functions, and `name` will be treated as static (see [Marking arguments as static](jit-compilation.html#jit-marking-arguments-as-static) for more information on static args):

    @jax.jit
    def f(x: MyDataclassContainer | MyOtherContainer):
      return x.a + x.b

    # Works fine! `mdc.name` is static.
    mdc = MyDataclassContainer('mdc', 1, 2, 3)
    y = f(mdc)

Contrast this with `MyOtherContainer`, the `NamedTuple` subclass. Since the `name` field is a pytree leaf, JIT expects it to be convertible to [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array"), and the following raises an error:

    moc = MyOtherContainer('moc', 1, 2, 3)
    y = f(moc)

    ---------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    Cell In[9], line 2
          1 moc = MyOtherContainer('moc', 1, 2, 3)
    ----> 2 y = f(moc)

        [... skipping hidden 3 frame]

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/pjit.py:650, in _infer_input_type(fun, dbg_fn, explicit_args)
        648   dbg = dbg_fn()
        649   arg_description = f"path {dbg.arg_names[i] if dbg.arg_names is not None else 'unknown'}"
    --> 650   raise TypeError(
        651     f"Error interpreting argument to {fun} as an abstract array."
        652     f" The problematic value is of type {type(x)} and was passed to"
        653     f" the function at {arg_description}.\n"
        654     "This typically means that a jit-wrapped function was called with a non-array"
        655     " argument, and this argument was not marked as static using the"
        656     " static_argnums or static_argnames parameters of jax.jit."
        657   ) from None
        658 if config.mutable_array_checks.value:
        659   check_no_aliased_ref_args(dbg_fn, avals, explicit_args)

    TypeError: Error interpreting argument to <function f at 0x707572142b60> as an abstract array. The problematic value is of type <class 'str'> and was passed to the function at path x.name.
    This typically means that a jit-wrapped function was called with a non-array argument, and this argument was not marked as static using the static_argnums or static_argnames parameters of jax.jit.

The whole set of functions for operating on pytrees are in [`jax.tree_util`](jax.tree_util.html#module-jax.tree_util "jax.tree_util").

## Custom pytrees and initialization with unexpected values[\#](#custom-pytrees-and-initialization-with-unexpected-values "Link to this heading")

Another common gotcha with user-defined pytree objects is that JAX transformations occasionally initialize them with unexpected values, so that any input validation done at initialization may fail. For example:

    class MyTree:
      def __init__(self, a):
        self.a = jnp.asarray(a)

    register_pytree_node(MyTree, lambda tree: ((tree.a,), None),
        lambda _, args: MyTree(*args))

    tree = MyTree(jnp.arange(5.0))

    jax.vmap(lambda x: x)(tree)      # Error because object() is passed to `MyTree`.

    <__main__.MyTree at 0x7075781b1130>

    jax.jacobian(lambda x: x)(tree)  # Error because MyTree(...) is passed to `MyTree`.

    ---------------------------------------------------------------------------
    ValueError                                Traceback (most recent call last)
    Cell In[11], line 1
    ----> 1 jax.jacobian(lambda x: x)(tree)  # Error because MyTree(...) is passed to `MyTree`.

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/api.py:809, in jacrev.<locals>.jacfun(*args, **kwargs)
        806 @wraps(fun, docstr=docstr, argnums=argnums)
        807 def jacfun(*args, **kwargs):
        808   f_partial, dyn_args = argnums_partial2(fun, argnums, args, kwargs)
    --> 809   tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
        810   y, pullback, *maybe_aux = vjp(f_partial, *dyn_args, has_aux=has_aux)
        811   tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/tree_util.py:399, in tree_map(f, tree, is_leaf, *rest)
        397   err = next(_prefix_error((), tree, r2, is_leaf), None)  # type: ignore
        398   raise (err('tree_map tree') if err is not None else e) from None
    --> 399 return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))

    Cell In[10], line 6, in <lambda>(_, args)
    ----> 6     lambda _, args: MyTree(*args))

    Cell In[10], line 3, in MyTree.__init__(self, a)
          2   def __init__(self, a):
    ----> 3     self.a = jnp.asarray(a)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/numpy/array_constructors.py:453, in asarray(a, dtype, order, copy, device, out_sharding)
        451 if dtype is not None:
        452   dtype = dtypes.check_and_canonicalize_user_dtype(dtype, "asarray")
    --> 453 return array(a, dtype=dtype, copy=bool(copy), order=order, device=device,
        454              out_sharding=out_sharding)

    File ~/checkouts/readthedocs.org/user_builds/jax/envs/latest/lib/python3.12/site-packages/jax/_src/numpy/array_constructors.py:285, in array(object, dtype, copy, order, ndmin, device, out_sharding, *args)
        282 leaves, treedef = tree_util.tree_flatten(
        283     object, is_leaf=lambda x: not isinstance(x, (list, tuple)))
        284 if any(leaf is None for leaf in leaves):
    --> 285   raise ValueError("None is not a valid value for jnp.array")
        286 leaves = [
        287     leaf
        288     if (leaf_jax_array := getattr(leaf, "__jax_array__", None)) is None
        289     else leaf_jax_array()
        290     for leaf in leaves
        291 ]
        292 if dtype is None:
        293   # Use lattice_result_type rather than result_type to avoid canonicalization.
        294   # Otherwise, weakly-typed inputs would have their dtypes canonicalized.

    ValueError: None is not a valid value for jnp.array

- In the first case with `jax.vmap(...)(tree)`, JAX’s internals use arrays of `object()` values to infer the structure of the tree

- In the second case with `jax.jacobian(...)(tree)`, the Jacobian of a function mapping a tree to a tree is defined as a tree of trees.

**Potential solution 1:**

- The `__init__` and `__new__` methods of custom pytree classes should generally avoid doing any array conversion or other input validation, or else anticipate and handle these special cases. For example:

    class MyTree:
      def __init__(self, a):
        if not (type(a) is object or a is None or isinstance(a, MyTree)):
          a = jnp.asarray(a)
        self.a = a

**Potential solution 2:**

- Structure your custom `tree_unflatten` function so that it avoids calling `__init__`. If you choose this route, make sure that your `tree_unflatten` function stays in sync with `__init__` if and when the code is updated. Example:

    def tree_unflatten(aux_data, children):
      del aux_data  # Unused in this class.
      obj = object.__new__(MyTree)
      obj.a = children[0]
      return obj

## Internal pytree handling[\#](#internal-pytree-handling "Link to this heading")

JAX flattens pytrees into lists of leaves at the `api.py` boundary (and also in control flow primitives). This keeps downstream JAX internals simpler: transformations like [`grad()`](_autosummary/jax.grad.html#jax.grad "jax.grad"), [`jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit"), and [`vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") can handle user functions that accept and return the myriad different Python containers, while all the other parts of the system can operate on functions that only take (multiple) array arguments and always return a flat list of arrays.

When JAX flattens a pytree it will produce a list of leaves and a `treedef` object that encodes the structure of the original value. The `treedef` can then be used to construct a matching structured value after transforming the leaves. Pytrees are tree-like, rather than DAG-like or graph-like, in that we handle them assuming referential transparency and that they can’t contain reference cycles.

Here is a simple example:

    from jax.tree_util import tree_flatten, tree_unflatten
    import jax.numpy as jnp

    # The structured value to be transformed
    value_structured = [1., (2., 3.)]

    # The leaves in value_flat correspond to the `*` markers in value_tree
    value_flat, value_tree = tree_flatten(value_structured)
    print(f"{value_flat=}\n{value_tree=}")

    # Transform the flat value list using an element-wise numeric transformer
    transformed_flat = list(map(lambda v: v * 2., value_flat))
    print(f"{transformed_flat=}")

    # Reconstruct the structured output, using the original
    transformed_structured = tree_unflatten(value_tree, transformed_flat)
    print(f"{transformed_structured=}")

    value_flat=[1.0, 2.0, 3.0]
    value_tree=PyTreeDef([*, (*, *)])
    transformed_flat=[2.0, 4.0, 6.0]
    transformed_structured=[2.0, (4.0, 6.0)]

By default, pytree containers can be lists, tuples, dicts, namedtuple, None, OrderedDict. Other types of values, including numeric and ndarray values, are treated as leaves:

    from collections import namedtuple
    Point = namedtuple('Point', ['x', 'y'])

    example_containers = [
        (1., [2., 3.]),
        (1., {'b': 2., 'a': 3.}),
        1.,
        None,
        jnp.zeros(2),
        Point(1., 2.)
    ]
    def show_example(structured):
      flat, tree = tree_flatten(structured)
      unflattened = tree_unflatten(tree, flat)
      print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")

    for structured in example_containers:
      show_example(structured)

    structured=(1.0, [2.0, 3.0])
      flat=[1.0, 2.0, 3.0]
      tree=PyTreeDef((*, [*, *]))
      unflattened=(1.0, [2.0, 3.0])
    structured=(1.0, {'b': 2.0, 'a': 3.0})
      flat=[1.0, 3.0, 2.0]
      tree=PyTreeDef((*, {'a': *, 'b': *}))
      unflattened=(1.0, {'a': 3.0, 'b': 2.0})
    structured=1.0
      flat=[1.0]
      tree=PyTreeDef(*)
      unflattened=1.0
    structured=None
      flat=[]
      tree=PyTreeDef(None)
      unflattened=None
    structured=Array([0., 0.], dtype=float32)
      flat=[Array([0., 0.], dtype=float32)]
      tree=PyTreeDef(*)
      unflattened=Array([0., 0.], dtype=float32)
    structured=Point(x=1.0, y=2.0)
      flat=[1.0, 2.0]
      tree=PyTreeDef(CustomNode(namedtuple[Point], [*, *]))
      unflattened=Point(x=1.0, y=2.0)

[](transfer_guard.html "previous page")

previous

Transfer guard

[](persistent_compilation_cache.html "next page")

next

Persistent compilation cache

Contents

- [Custom pytrees and initialization with unexpected values](#custom-pytrees-and-initialization-with-unexpected-values)
- [Internal pytree handling](#internal-pytree-handling)

By The JAX authors

© Copyright 2024, The JAX Authors.\
