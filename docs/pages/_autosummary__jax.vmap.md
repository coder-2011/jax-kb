- [](../index.html)
- [API Reference](../jax.html)
- jax.vmap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.vmap.rst "Download source file")
-  .pdf

# jax.vmap

## Contents

- [`vmap()`](#jax.vmap)

# jax.vmap[\#](#jax-vmap "Link to this heading")

jax.vmap(*fun*, *in_axes=0*, *out_axes=0*, *axis_name=None*, *axis_size=None*, *spmd_axis_name=None*, *sum_match=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L974-L1206)[\#](#jax.vmap "Link to this definition")  
Vectorizing map. Creates a function which maps `fun` over argument axes.

Parameters:  
- **fun** (*F*) – Function to be mapped over additional axes.

- **in_axes** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None* *\|* *Sequence\[Any\]*) –

  An integer, None, or sequence of values specifying which input array axes to map over.

  If each positional argument to `fun` is an array, then `in_axes` can be an integer, a None, or a tuple of integers and Nones with length equal to the number of positional arguments to `fun`. An integer or `None` indicates which array axis to map over for all arguments (with `None` indicating not to map any axis), and a tuple indicates which axis to map for each corresponding positional argument. Axis integers must be in the range `[-ndim,`` ``ndim)` for each array, where `ndim` is the number of dimensions (axes) of the corresponding input array.

  If the positional arguments to `fun` are container (pytree) types, `in_axes` must be a sequence with length equal to the number of positional arguments to `fun`, and for each argument the corresponding element of `in_axes` can be a container with a matching pytree structure specifying the mapping of its container elements. In other words, `in_axes` must be a container tree prefix of the positional argument tuple passed to `fun`. See this link for more detail: [https://docs.jax.dev/en/latest/pytrees.html#applying-optional-parameters-to-pytrees](https://docs.jax.dev/en/latest/pytrees.html#applying-optional-parameters-to-pytrees)

  Either `axis_size` must be provided explicitly, or at least one positional argument must have `in_axes` not None. The sizes of the mapped input axes for all mapped positional arguments must all be equal.

  Arguments passed as keywords are always mapped over their leading axis (i.e. axis index 0).

  See below for examples.

- **out_axes** (*Any*) – An integer, None, or (nested) standard Python container (tuple/list/dict) thereof indicating where the mapped axis should appear in the output. All outputs with a mapped axis must have a non-None `out_axes` specification. Axis integers must be in the range `[-ndim,`` ``ndim)` for each output array, where `ndim` is the number of dimensions (axes) of the array returned by the [`vmap()`](#jax.vmap "jax.vmap")-ed function, which is one more than the number of dimensions (axes) of the corresponding array returned by `fun`.

- **axis_name** (*AxisName* *\|* *None*) – Optional, a hashable Python object used to identify the mapped axis so that parallel collectives can be applied.

- **axis_size** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *None*) – Optional, an integer indicating the size of the axis to be mapped. If not provided, the mapped axis size is inferred from arguments.

- **spmd_axis_name** (*AxisName* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[AxisName,* *...\]* *\|* *None*)

- **sum_match** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)"))

Returns:  
Batched/vectorized version of `fun` with arguments that correspond to those of `fun`, but with extra array axes at positions indicated by `in_axes`, and a return value that corresponds to that of `fun`, but with extra array axes at positions indicated by `out_axes`.

Return type:  
F

For example, we can implement a matrix-matrix product using a vector dot product:

    >>> import jax.numpy as jnp
    >>>
    >>> vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
    >>> mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
    >>> mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

Here we use `[a,b]` to indicate an array with shape (a,b). Here are some variants:

    >>> mv1 = vmap(vv, (0, 0), 0)   #  ([b,a], [b,a]) -> [b]        (b is the mapped axis)
    >>> mv2 = vmap(vv, (0, 1), 0)   #  ([b,a], [a,b]) -> [b]        (b is the mapped axis)
    >>> mm2 = vmap(mv2, (1, 1), 0)  #  ([b,c,a], [a,c,b]) -> [c,b]  (c is the mapped axis)

Here’s an example of using container types in `in_axes` to specify which axes of the container elements to map over:

    >>> A, B, C, D = 2, 3, 4, 5
    >>> x = jnp.ones((A, B))
    >>> y = jnp.ones((B, C))
    >>> z = jnp.ones((C, D))
    >>> def foo(tree_arg):
    ...   x, (y, z) = tree_arg
    ...   return jnp.dot(x, jnp.dot(y, z))
    >>> tree = (x, (y, z))
    >>> print(foo(tree))
    [[12. 12. 12. 12. 12.]
     [12. 12. 12. 12. 12.]]
    >>> from jax import vmap
    >>> K = 6  # batch size
    >>> x = jnp.ones((K, A, B))  # batch axis in different locations
    >>> y = jnp.ones((B, K, C))
    >>> z = jnp.ones((C, D, K))
    >>> tree = (x, (y, z))
    >>> vfoo = vmap(foo, in_axes=((0, (1, 2)),))
    >>> print(vfoo(tree).shape)
    (6, 2, 5)

Here’s another example using container types in `in_axes`, this time a dictionary, to specify the elements of the container to map over:

    >>> dct = {'a': 0., 'b': jnp.arange(5.)}
    >>> x = 1.
    >>> def foo(dct, x):
    ...  return dct['a'] + dct['b'] + x
    >>> out = vmap(foo, in_axes=({'a': None, 'b': 0}, None))(dct, x)
    >>> print(out)
    [1. 2. 3. 4. 5.]

The results of a vectorized function can be mapped or unmapped. For example, the function below returns a pair with the first element mapped and the second unmapped. Only for unmapped results we can specify `out_axes` to be `None` (to keep it unmapped).

    >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=(0, None))(jnp.arange(2.), 4.))
    (Array([4., 5.], dtype=float32), 8.0)

If the `out_axes` is specified for an unmapped result, the result is broadcast across the mapped axis:

    >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=0)(jnp.arange(2.), 4.))
    (Array([4., 5.], dtype=float32), Array([8., 8.], dtype=float32, weak_type=True))

If the `out_axes` is specified for a mapped result, the result is transposed accordingly.

Finally, here’s an example using `axis_name` together with collectives:

    >>> xs = jnp.arange(3. * 4.).reshape(3, 4)
    >>> print(vmap(lambda x: lax.psum(x, 'i'), axis_name='i')(xs))
    [[12. 15. 18. 21.]
     [12. 15. 18. 21.]
     [12. 15. 18. 21.]]

See the [`jax.pmap()`](jax.pmap.html#jax.pmap "jax.pmap") docstring for more examples involving collectives.

[](jax.checkpoint.html "previous page")

previous

jax.checkpoint

[](jax.shard_map.html "next page")

next

jax.shard_map

Contents

- [`vmap()`](#jax.vmap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
