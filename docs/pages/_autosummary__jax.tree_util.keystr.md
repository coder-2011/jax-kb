- [](../index.html)
- [API Reference](../jax.html)
- [`jax.tree_util` module](../jax.tree_util.html)
- jax.tree_util.keystr

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.tree_util.keystr.rst "Download source file")
-  .pdf

# jax.tree_util.keystr

## Contents

- [`keystr()`](#jax.tree_util.keystr)

# jax.tree_util.keystr[\#](#jax-tree-util-keystr "Link to this heading")

jax.tree_util.keystr(*keys*, *\**, *simple=False*, *separator=''*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/tree_util.py#L833-L864)[\#](#jax.tree_util.keystr "Link to this definition")  
Helper to pretty-print a tuple of keys.

Parameters:  
- **keys** (*KeyPath*) – A tuple of `KeyEntry` or any class that can be converted to string.

- **simple** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, use a simplified string representation for keys. The simple representation of keys will be more compact than the default, but is ambiguous in some cases (for example “0” might refer to the first item in a list or a dictionary key for the integer 0 or string “0”).

- **separator** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The separator to use to join string representations of the keys.

Returns:  
A string that joins all string representations of the keys.

Return type:  
[str](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")

Examples

    >>> import jax
    >>> params = {'foo': {'bar': {'baz': 1, 'bat': [2, 3]}}}
    >>> for path, _ in jax.tree_util.tree_leaves_with_path(params):
    ...   print(jax.tree_util.keystr(path))
    ['foo']['bar']['bat'][0]
    ['foo']['bar']['bat'][1]
    ['foo']['bar']['baz']
    >>> for path, _ in jax.tree_util.tree_leaves_with_path(params):
    ...   print(jax.tree_util.keystr(path, simple=True, separator='/'))
    foo/bar/bat/0
    foo/bar/bat/1
    foo/bar/baz

[](jax.tree_util.KeyPath.html "previous page")

previous

jax.tree_util.KeyPath

[](jax.tree_util.tree_all.html "next page")

next

jax.tree_util.tree_all

Contents

- [`keystr()`](#jax.tree_util.keystr)

By The JAX authors

© Copyright 2024, The JAX Authors.\
