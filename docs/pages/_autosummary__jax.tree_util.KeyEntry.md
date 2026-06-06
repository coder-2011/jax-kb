- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.KeyEntry

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.KeyEntry.rst "Download source file")
-  .pdf

# jax.tree_util.KeyEntry

## Contents

- [`KeyEntry`](#jax.tree_util.KeyEntry)

# jax.tree_util.KeyEntry[\#](#jax-tree-util-keyentry "Link to this heading")

jax.tree_util.KeyEntry *= ~KeyEntry*[\#](#jax.tree_util.KeyEntry "Link to this definition")  
Type variable.

The preferred way to construct a type variable is via the dedicated syntax for generic functions, classes, and type aliases:

    class Sequence[T]:  # T is a TypeVar
        ...

This syntax can also be used to create bound and constrained type variables:

    # S is a TypeVar bound to str
    class StrSequence[S: str]:
        ...

    # A is a TypeVar constrained to str or bytes
    class StrOrBytesSequence[A: (str, bytes)]:
        ...

However, if desired, reusable type variables can also be constructed manually, like so:

    T = TypeVar('T')  # Can be anything
    S = TypeVar('S', bound=str)  # Can be any subtype of str
    A = TypeVar('A', str, bytes)  # Must be exactly str or bytes

Type variables exist primarily for the benefit of static type checkers. They serve as the parameters for generic types as well as for generic function and type alias definitions.

The variance of type variables is inferred by type checkers when they are created through the type parameter syntax and when `infer_variance=True` is passed. Manually created type variables may be explicitly marked covariant or contravariant by passing `covariant=True` or `contravariant=True`. By default, manually created type variables are invariant. See PEP 484 and PEP 695 for more details.

[](jax.tree_util.treedef_tuple.html "previous page")

previous

jax.tree_util.treedef_tuple

[](jax.tree_util.KeyPath.html "next page")

next

jax.tree_util.KeyPath

Contents

- [`KeyEntry`](#jax.tree_util.KeyEntry)

By The JAX authors

© Copyright 2024, The JAX Authors.\
