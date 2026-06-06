- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.register_static

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.register_static.rst "Download source file")
-  .pdf

# jax.tree_util.register_static

## Contents

- [`register_static()`](#jax.tree_util.register_static)

# jax.tree_util.register_static[\#](#jax-tree-util-register-static "Link to this heading")

jax.tree_util.register_static(*cls*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L1190-L1226)[\#](#jax.tree_util.register_static "Link to this definition")  
Registers cls as a pytree with no leaves.

Instances are treated as static by [`jax.jit()`](jax.jit.html#jax.jit "jax.jit"), [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap"), etc. This can be an alternative to labeling inputs as static using `jit`’s `static_argnums` and `static_argnames` kwargs, `pmap`’s `static_broadcasted_argnums`, etc.

Parameters:  
**cls** ([*type*](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")*\[H\]*) – type to be registered as static. Must be hashable, as defined in [https://docs.python.org/3/glossary.html#term-hashable](https://docs.python.org/3/glossary.html#term-hashable).

Returns:  
The input class `cls` is returned unchanged after being added to JAX’s pytree registry. This allows `register_static` to be used as a decorator.

Return type:  
[type](https://docs.python.org/3/library/functions.html#type "(in Python v3.14)")\[H\]

Examples

    >>> import jax
    >>> @jax.tree_util.register_static
    ... class StaticStr(str):
    ...   pass

This static string can now be used directly in [`jax.jit()`](jax.jit.html#jax.jit "jax.jit")-compiled functions, without marking the variable static using `static_argnums`:

    >>> @jax.jit
    ... def f(x, y, s):
    ...   return x + y if s == 'add' else x - y
    ...
    >>> f(1, 2, StaticStr('add'))
    Array(3, dtype=int32, weak_type=True)

[](jax.tree_util.register_pytree_with_keys_class.html "previous page")

previous

jax.tree_util.register_pytree_with_keys_class

[](jax.tree_util.tree_flatten_with_path.html "next page")

next

jax.tree_util.tree_flatten_with_path

Contents

- [`register_static()`](#jax.tree_util.register_static)

By The JAX authors

© Copyright 2024, The JAX Authors.\
