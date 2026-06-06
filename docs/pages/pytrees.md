- [](index.html)
- [JAX 101](jax-101.html)
- Pytrees

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .md](_sources/pytrees.md "Download source file")
-  .pdf

# Pytrees

## Contents

- [What is a pytree?](#what-is-a-pytree)
- [Common pytree functions](#common-pytree-functions)
  - [Common function: `jax.tree.map`](#common-function-jax-tree-map)
  - [Example of `jax.tree.map` with ML model parameters](#example-of-jax-tree-map-with-ml-model-parameters)
- [Viewing the pytree definition of an object](#viewing-the-pytree-definition-of-an-object)
- [Pytrees and JAX transformations](#pytrees-and-jax-transformations)
- [Explicit key paths](#explicit-key-paths)
- [Common pytree gotchas](#common-pytree-gotchas)
  - [Mistaking pytree nodes for leaves](#mistaking-pytree-nodes-for-leaves)
  - [Handling of `None` by `jax.tree_util`](#handling-of-none-by-jax-tree-util)
  - [Dictionary keys must be sortable](#dictionary-keys-must-be-sortable)
- [Common pytree patterns](#common-pytree-patterns)
  - [Transposing pytrees with `jax.tree.map` and `jax.tree.transpose`](#transposing-pytrees-with-jax-tree-map-and-jax-tree-transpose)
  - [Extending pytrees](#extending-pytrees)

# Pytrees[\#](#working-with-pytrees "Link to this heading")

JAX has built-in support for objects that look like dictionaries (dicts) of arrays, or lists of lists of dicts, or other nested structures — in JAX these are called pytrees. This section will explain how to use them, provide useful code examples, and point out common “gotchas” and patterns.

For an explanation of how to create custom pytrees, see [Custom pytree nodes](custom_pytrees.html).

## What is a pytree?[\#](#what-is-a-pytree "Link to this heading")

A pytree is a container-like structure built out of container-like Python objects — “leaf” pytrees and/or more pytrees. A pytree can include lists, tuples, and dicts. A leaf is anything that’s not a pytree, such as an array, but a single leaf is also a pytree.

In the context of machine learning (ML), a pytree can contain:

- Model parameters

- Dataset entries

- Reinforcement learning agent observations

When working with datasets, you can often come across pytrees (such as lists of lists of dicts).

Below is an example of a simple pytree. In JAX, you can use [`jax.tree.leaves()`](_autosummary/jax.tree.leaves.html#jax.tree.leaves "jax.tree.leaves"), to extract the flattened leaves from the trees, as demonstrated here:

    import jax
    import jax.numpy as jnp

    example_trees = [
        [1, 'a', object()],
        (1, (2, 3), ()),
        [1, {'k1': 2, 'k2': (3, 4)}, 5],
        {'a': 2, 'b': (2, 3)},
        jnp.array([1, 2, 3]),
    ]

    # Print how many leaves the pytrees have.
    for pytree in example_trees:
      # This `jax.tree.leaves()` method extracts the flattened leaves from the pytrees.
      leaves = jax.tree.leaves(pytree)
      print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}")

    [1, 'a', <object object at 0x7db586968850>]   has 3 leaves: [1, 'a', <object object at 0x7db586968850>]
    (1, (2, 3), ())                               has 3 leaves: [1, 2, 3]
    [1, {'k1': 2, 'k2': (3, 4)}, 5]               has 5 leaves: [1, 2, 3, 4, 5]
    {'a': 2, 'b': (2, 3)}                         has 3 leaves: [2, 2, 3]
    Array([1, 2, 3], dtype=int32)                 has 1 leaves: [Array([1, 2, 3], dtype=int32)]

Any tree-like structure built out of container-like Python objects can be treated as a pytree in JAX. Classes are considered container-like if they are in the pytree registry, which by default includes lists, tuples, and dicts. Any object whose type is *not* in the pytree container registry will be treated as a leaf node in the tree.

The pytree registry can be extended to include user-defined container classes by registering the class with functions that specify how to flatten the tree; see [Custom pytree nodes](custom_pytrees.html#pytrees-custom-pytree-nodes) below.

## Common pytree functions[\#](#common-pytree-functions "Link to this heading")

JAX provides a number of utilities to operate over pytrees. These can be found in the [`jax.tree_util`](jax.tree_util.html#module-jax.tree_util "jax.tree_util") subpackage; for convenience many of these have aliases in the [`jax.tree`](jax.tree.html#module-jax.tree "jax.tree") module.

### Common function: `jax.tree.map`[\#](#common-function-jax-tree-map "Link to this heading")

The most commonly used pytree function is [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map"). It works analogously to Python’s native `map`, but transparently operates over entire pytrees.

Here’s an example:

    list_of_lists = [
        [1, 2, 3],
        [1, 2],
        [1, 2, 3, 4]
    ]

    jax.tree.map(lambda x: x*2, list_of_lists)

    [[2, 4, 6], [2, 4], [2, 4, 6, 8]]

[`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") also allows mapping a [N-ary](https://en.wikipedia.org/wiki/N-ary) function over multiple arguments. For example:

    another_list_of_lists = list_of_lists
    jax.tree.map(lambda x, y: x+y, list_of_lists, another_list_of_lists)

    [[2, 4, 6], [2, 4], [2, 4, 6, 8]]

When using multiple arguments with [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map"), the structure of the inputs must exactly match. That is, lists must have the same number of elements, dicts must have the same keys, etc.

### Example of `jax.tree.map` with ML model parameters[\#](#example-of-jax-tree-map-with-ml-model-parameters "Link to this heading")

This example demonstrates how pytree operations can be useful when training a simple [multi-layer perceptron (MLP)](https://en.wikipedia.org/wiki/Multilayer_perceptron).

Begin with defining the initial model parameters:

    import numpy as np

    def init_mlp_params(layer_widths):
      params = []
      for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(weights=np.random.normal(size=(n_in, n_out)) * np.sqrt(2/n_in),
                 biases=np.ones(shape=(n_out,))
                )
        )
      return params

    params = init_mlp_params([1, 128, 128, 1])

Use [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") to check the shapes of the initial parameters:

    jax.tree.map(lambda x: x.shape, params)

    [{'biases': (128,), 'weights': (1, 128)},
     {'biases': (128,), 'weights': (128, 128)},
     {'biases': (1,), 'weights': (128, 1)}]

Next, define the functions for training the MLP model:

    # Define the forward pass.
    def forward(params, x):
      *hidden, last = params
      for layer in hidden:
        x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
      return x @ last['weights'] + last['biases']

    # Define the loss function.
    def loss_fn(params, x, y):
      return jnp.mean((forward(params, x) - y) ** 2)

    # Set the learning rate.
    LEARNING_RATE = 0.0001

    # Using the stochastic gradient descent, define the parameter update function.
    # Apply `@jax.jit` for JIT compilation (speed).
    @jax.jit
    def update(params, x, y):
      # Calculate the gradients with `jax.grad`.
      grads = jax.grad(loss_fn)(params, x, y)
      # Note that `grads` is a pytree with the same structure as `params`.
      # `jax.grad` is one of many JAX functions that has
      # built-in support for pytrees.
      # This is useful - you can apply the SGD update using JAX pytree utilities.
      return jax.tree.map(
          lambda p, g: p - LEARNING_RATE * g, params, grads
      )

## Viewing the pytree definition of an object[\#](#viewing-the-pytree-definition-of-an-object "Link to this heading")

To view the pytree definition of an arbitrary `object` for debugging purposes, you can use:

    from jax.tree_util import tree_structure
    print(tree_structure(object))

    PyTreeDef(*)

## Pytrees and JAX transformations[\#](#pytrees-and-jax-transformations "Link to this heading")

Many JAX functions, like [`jax.lax.scan()`](_autosummary/jax.lax.scan.html#jax.lax.scan "jax.lax.scan"), operate over pytrees of arrays. In addition, all JAX function transformations can be applied to functions that accept as input and produce as output pytrees of arrays.

Some JAX function transformations take optional parameters that specify how certain input or output values should be treated (such as the `in_axes` and `out_axes` arguments to [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap")). These parameters can also be pytrees, and their structure must correspond to the pytree structure of the corresponding arguments. In particular, to be able to “match up” leaves in these parameter pytrees with values in the argument pytrees, the parameter pytrees are often constrained to be tree prefixes of the argument pytrees.

For example, if you pass the following input to [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") (note that the input arguments to a function are considered a tuple):

    vmap(f, in_axes=(a1, {"k1": a2, "k2": a3}))

then you can use the following `in_axes` pytree to specify that only the `k2` argument is mapped (`axis=0`), and the rest aren’t mapped over (`axis=None`):

    vmap(f, in_axes=(None, {"k1": None, "k2": 0}))

The optional parameter pytree structure must match that of the main input pytree. However, the optional parameters can optionally be specified as a “prefix” pytree, meaning that a single leaf value can be applied to an entire sub-pytree.

For example, if you have the same [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap") input as above, but wish to only map over the dictionary argument, you can use:

    vmap(f, in_axes=(None, 0))  # equivalent to (None, {"k1": 0, "k2": 0})

Alternatively, if you want every argument to be mapped, you can write a single leaf value that is applied over the entire argument tuple pytree:

    vmap(f, in_axes=0)  # equivalent to (0, {"k1": 0, "k2": 0})

This happens to be the default `in_axes` value for [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap").

The same logic applies to other optional parameters that refer to specific input or output values of a transformed function, such as `out_axes` in [`jax.vmap()`](_autosummary/jax.vmap.html#jax.vmap "jax.vmap").

## Explicit key paths[\#](#explicit-key-paths "Link to this heading")

In a pytree each leaf has a *key path*. A key path for a leaf is a `list` of *keys*, where the length of the list is equal to the depth of the leaf in the pytree . Each *key* is a [hashable object](https://docs.python.org/3/glossary.html#term-hashable) that represents an index into the corresponding pytree node type. The type of the key depends on the pytree node type; for example, the type of keys for `dict`s is different from the type of keys for `tuple`s.

For built-in pytree node types, the set of keys for any pytree node instance is unique. For a pytree comprising nodes with this property, the key path for each leaf is unique.

JAX has the following `jax.tree_util.*` methods for working with key paths:

- [`jax.tree_util.tree_flatten_with_path()`](_autosummary/jax.tree_util.tree_flatten_with_path.html#jax.tree_util.tree_flatten_with_path "jax.tree_util.tree_flatten_with_path"): Works similarly to [`jax.tree.flatten()`](_autosummary/jax.tree.flatten.html#jax.tree.flatten "jax.tree.flatten"), but returns key paths.

- [`jax.tree_util.tree_map_with_path()`](_autosummary/jax.tree_util.tree_map_with_path.html#jax.tree_util.tree_map_with_path "jax.tree_util.tree_map_with_path"): Works similarly to [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map"), but the function also takes key paths as arguments.

- [`jax.tree_util.keystr()`](_autosummary/jax.tree_util.keystr.html#jax.tree_util.keystr "jax.tree_util.keystr"): Given a general key path, returns a reader-friendly string expression.

For example, one use case is to print debugging information related to a certain leaf value:

    import collections

    ATuple = collections.namedtuple("ATuple", ('name'))

    tree = [1, {'k1': 2, 'k2': (3, 4)}, ATuple('foo')]
    flattened, _ = jax.tree_util.tree_flatten_with_path(tree)

    for key_path, value in flattened:
      print(f'Value of tree{jax.tree_util.keystr(key_path)}: {value}')

    Value of tree[0]: 1
    Value of tree[1]['k1']: 2
    Value of tree[1]['k2'][0]: 3
    Value of tree[1]['k2'][1]: 4
    Value of tree[2].name: foo

To express key paths, JAX provides a few default key types for the built-in pytree node types, namely:

- `SequenceKey(idx:`` ``int)`: For lists and tuples.

- `DictKey(key:`` ``Hashable)`: For dictionaries.

- `GetAttrKey(name:`` ``str)`: For `namedtuple`s and preferably custom pytree nodes (more in the next section)

You are free to define your own key types for your custom nodes. They will work with [`jax.tree_util.keystr()`](_autosummary/jax.tree_util.keystr.html#jax.tree_util.keystr "jax.tree_util.keystr") as long as their `__str__()` method is also overridden with a reader-friendly expression.

    for key_path, _ in flattened:
      print(f'Key path of tree{jax.tree_util.keystr(key_path)}: {repr(key_path)}')

    Key path of tree[0]: (SequenceKey(idx=0),)
    Key path of tree[1]['k1']: (SequenceKey(idx=1), DictKey(key='k1'))
    Key path of tree[1]['k2'][0]: (SequenceKey(idx=1), DictKey(key='k2'), SequenceKey(idx=0))
    Key path of tree[1]['k2'][1]: (SequenceKey(idx=1), DictKey(key='k2'), SequenceKey(idx=1))
    Key path of tree[2].name: (SequenceKey(idx=2), GetAttrKey(name='name'))

## Common pytree gotchas[\#](#common-pytree-gotchas "Link to this heading")

This section covers some of the most common problems (“gotchas”) encountered when using JAX pytrees.

### Mistaking pytree nodes for leaves[\#](#mistaking-pytree-nodes-for-leaves "Link to this heading")

A common gotcha to look out for is accidentally introducing *tree nodes* instead of *leaves*:

    a_tree = [jnp.zeros((2, 3)), jnp.zeros((3, 4))]

    # Try to make another pytree with ones instead of zeros.
    shapes = jax.tree.map(lambda x: x.shape, a_tree)
    jax.tree.map(jnp.ones, shapes)

    [(Array([1., 1.], dtype=float32), Array([1., 1., 1.], dtype=float32)),
     (Array([1., 1., 1.], dtype=float32), Array([1., 1., 1., 1.], dtype=float32))]

What happened here is that the `shape` of an array is a tuple, which is a pytree node, with its elements as leaves. Thus, in the map, instead of calling `jnp.ones` on e.g. `(2,`` ``3)`, it’s called on `2` and `3`.

The solution will depend on the specifics, but there are two broadly applicable options:

- Rewrite the code to avoid the intermediate [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map").

- Convert the tuple into a NumPy array (`np.array`) or a JAX NumPy array (`jnp.array`), which makes the entire sequence a leaf.

### Handling of `None` by `jax.tree_util`[\#](#handling-of-none-by-jax-tree-util "Link to this heading")

`jax.tree_util` functions treat `None` as the absence of a pytree node, not as a leaf:

    jax.tree.leaves([None, None, None])

    []

To treat `None` as a leaf, you can use the `is_leaf` argument:

    jax.tree.leaves([None, None, None], is_leaf=lambda x: x is None)

    [None, None, None]

### Dictionary keys must be sortable[\#](#dictionary-keys-must-be-sortable "Link to this heading")

`jax.tree_util` flattens a dictionary by its sorted keys, so the resulting pytree structure depends only on the *set* of keys and not on the insertion order. This keeps structure equality — and therefore JIT caching — predictable, but it means a dictionary’s keys must be sortable relative to one another. Mixing key types that have no ordering between them, such as `int` and `str`, raises an error:

    jax.tree.map(lambda x: x + 1, {1: 7, "y": 42})

    ValueError: Comparator raised exception while sorting pytree dictionary keys.

If you need a mapping whose keys can’t be ordered, you can use `collections.OrderedDict`, which JAX flattens in insertion order without sorting the keys, or register a custom pytree node that defines its own flattening and unflattening (see [Custom pytree nodes](custom_pytrees.html)).

## Common pytree patterns[\#](#common-pytree-patterns "Link to this heading")

This section covers some of the most common patterns with JAX pytrees.

### Transposing pytrees with `jax.tree.map` and `jax.tree.transpose`[\#](#transposing-pytrees-with-jax-tree-map-and-jax-tree-transpose "Link to this heading")

To transpose a pytree (turn a list of trees into a tree of lists), JAX has two functions: [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map") (more basic) and [`jax.tree.transpose()`](_autosummary/jax.tree.transpose.html#jax.tree.transpose "jax.tree.transpose") (more flexible, complex and verbose).

**Option 1:** Use [`jax.tree.map()`](_autosummary/jax.tree.map.html#jax.tree.map "jax.tree.map"). Here’s an example:

    def tree_transpose(list_of_trees):
      """
      Converts a list of trees of identical structure into a single tree of lists.
      """
      return jax.tree.map(lambda *xs: list(xs), *list_of_trees)

    # Convert a dataset from row-major to column-major.
    episode_steps = [dict(t=1, obs=3), dict(t=2, obs=4)]
    tree_transpose(episode_steps)

    {'obs': [3, 4], 't': [1, 2]}

**Option 2:** For more complex transposes, use [`jax.tree.transpose()`](_autosummary/jax.tree.transpose.html#jax.tree.transpose "jax.tree.transpose"), which is more verbose, but allows you specify the structure of the inner and outer pytree for more flexibility. For example:

    jax.tree.transpose(
      outer_treedef = jax.tree.structure([0 for e in episode_steps]),
      inner_treedef = jax.tree.structure(episode_steps[0]),
      pytree_to_transpose = episode_steps
    )

    {'obs': [3, 4], 't': [1, 2]}

### Extending pytrees[\#](#extending-pytrees "Link to this heading")

Material on extending pytrees has been moved to [Custom pytree nodes](custom_pytrees.html#pytrees-custom-pytree-nodes).

[](automatic-differentiation.html "previous page")

previous

Automatic differentiation

[](random-numbers.html "next page")

next

Pseudorandom numbers

Contents

- [What is a pytree?](#what-is-a-pytree)
- [Common pytree functions](#common-pytree-functions)
  - [Common function: `jax.tree.map`](#common-function-jax-tree-map)
  - [Example of `jax.tree.map` with ML model parameters](#example-of-jax-tree-map-with-ml-model-parameters)
- [Viewing the pytree definition of an object](#viewing-the-pytree-definition-of-an-object)
- [Pytrees and JAX transformations](#pytrees-and-jax-transformations)
- [Explicit key paths](#explicit-key-paths)
- [Common pytree gotchas](#common-pytree-gotchas)
  - [Mistaking pytree nodes for leaves](#mistaking-pytree-nodes-for-leaves)
  - [Handling of `None` by `jax.tree_util`](#handling-of-none-by-jax-tree-util)
  - [Dictionary keys must be sortable](#dictionary-keys-must-be-sortable)
- [Common pytree patterns](#common-pytree-patterns)
  - [Transposing pytrees with `jax.tree.map` and `jax.tree.transpose`](#transposing-pytrees-with-jax-tree-map-and-jax-tree-transpose)
  - [Extending pytrees](#extending-pytrees)

By The JAX authors

© Copyright 2024, The JAX Authors.\
