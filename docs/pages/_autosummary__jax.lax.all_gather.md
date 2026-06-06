- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.all_gather

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.all_gather.rst "Download source file")
-  .pdf

# jax.lax.all_gather

## Contents

- [`all_gather()`](#jax.lax.all_gather)

# jax.lax.all_gather[\#](#jax-lax-all-gather "Link to this heading")

jax.lax.all_gather(*x*, *axis_name*, *\**, *axis_index_groups=None*, *axis=0*, *tiled=False*, *to='varying'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/parallel.py#L1669-L1738)[\#](#jax.lax.all_gather "Link to this definition")  
Gather values of x across all replicas.

If `x` is a pytree then the result is equivalent to mapping this function to each leaf in the tree.

This is equivalent to, but faster than, all_to_all(broadcast(x)).

Parameters:  
- **x** – array(s) with a mapped axis named `axis_name`.

- **axis_name** – hashable Python object used to name a pmapped axis (see the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") documentation for more details).

- **axis_index_groups** – optional list of lists containing axis indices (e.g. for an axis of size 4, \[\[0, 1\], \[2, 3\]\] would run all gather over the first two and last two replicas). Groups must cover all axis indices exactly once, and all groups must be the same size.

- **axis** – a positional axis into which the chunks along `axis_name` will be concatenated.

- **tiled** – when `False`, the chunks will be stacked into a fresh positional axis at index `axis` in the output. When `True`, `axis` has to refer to an existing positional dimension and the chunks will be concatenated into that dimension.

- **to** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – The output manual axis type, defaults to ‘varying’. Valid options are: ‘varying’, ‘reduced’ and ‘invarying’.

Returns:  
Array(s) representing the result of an all-gather along the axis `axis_name`. Shapes are the same as `x.shape`, but:

- when `tiled` is `False`, there is a new dimension equal to the size of axis `axis_name` in position `axis`,

- when `tiled` is `True`, the size of dimension in position `axis` is multiplied by the size of axis `axis_name`.

For example, with 4 XLA devices available:

    >>> x = np.arange(4)
    >>> y = jax.pmap(lambda x: jax.lax.all_gather(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [[0 1 2 3]
     [0 1 2 3]
     [0 1 2 3]
     [0 1 2 3]]

An example of using axis_index_groups, groups split by even & odd device ids:

    >>> x = np.arange(16).reshape(4, 4)
    >>> print(x)
      [[ 0  1  2  3]
       [ 4  5  6  7]
       [ 8  9 10 11]
       [12 13 14 15]]
    >>> def f(x):
    ...   return jax.lax.all_gather(
    ...       x, 'i', axis_index_groups=[[0, 2], [3, 1]])
    >>> y = jax.pmap(f, axis_name='i')(x)
    >>> print(y)
    [[[ 0  1  2  3]
      [ 8  9 10 11]]
     [[12 13 14 15]
      [ 4  5  6  7]]
     [[ 0  1  2  3]
      [ 8  9 10 11]]
     [[12 13 14 15]
      [ 4  5  6  7]]]

[](jax.lax.custom_root.html "previous page")

previous

jax.lax.custom_root

[](jax.lax.all_to_all.html "next page")

next

jax.lax.all_to_all

Contents

- [`all_gather()`](#jax.lax.all_gather)

By The JAX authors

© Copyright 2024, The JAX Authors.\
