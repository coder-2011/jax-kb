- [](../index.html)
- [API Reference](../jax.html)
- jax.hessian

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.hessian.rst "Download source file")
-  .pdf

# jax.hessian

## Contents

- [`hessian()`](#jax.hessian)

# jax.hessian[\#](#jax-hessian "Link to this heading")

jax.hessian(*fun*, *argnums=0*, *has_aux=False*, *holomorphic=False*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/api.py#L832-L899)[\#](#jax.hessian "Link to this definition")  
Hessian of `fun` as a dense array.

Parameters:  
- **fun** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")) – Function whose Hessian is to be computed. Its arguments at positions specified by `argnums` should be arrays, scalars, or standard Python containers thereof. It should return arrays, scalars, or standard Python containers thereof.

- **argnums** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]*) – Optional, integer or sequence of integers. Specifies which positional argument(s) to differentiate with respect to (default `0`).

- **has_aux** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Indicates whether `fun` returns a pair where the first element is considered the output of the mathematical function to be differentiated and the second element is auxiliary data. Default False.

- **holomorphic** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – Optional, bool. Indicates whether `fun` is promised to be holomorphic. Default False.

Returns:  
A function with the same arguments as `fun`, that evaluates the Hessian of `fun`.

Return type:  
[Callable](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")

    >>> import jax
    >>>
    >>> g = lambda x: x[0]**3 - 2*x[0]*x[1] - x[1]**6
    >>> print(jax.hessian(g)(jax.numpy.array([1., 2.])))
    [[   6.   -2.]
     [  -2. -480.]]

[`hessian()`](#jax.hessian "jax.hessian") is a generalization of the usual definition of the Hessian that supports nested Python containers (i.e. pytrees) as inputs and outputs. The tree structure of `jax.hessian(fun)(x)` is given by forming a tree product of the structure of `fun(x)` with a tree product of two copies of the structure of `x`. A tree product of two tree structures is formed by replacing each leaf of the first tree with a copy of the second. For example:

    >>> import jax.numpy as jnp
    >>> f = lambda dct: {"c": jnp.power(dct["a"], dct["b"])}
    >>> print(jax.hessian(f)({"a": jnp.arange(2.) + 1., "b": jnp.arange(2.) + 2.}))
    {'c': {'a': {'a': Array([[[ 2.,  0.], [ 0.,  0.]],
                             [[ 0.,  0.], [ 0., 12.]]], dtype=float32),
                 'b': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                             [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32)},
           'b': {'a': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                             [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32),
                 'b': Array([[[0.      , 0.      ], [0.      , 0.      ]],
                             [[0.      , 0.      ], [0.      , 3.843624]]], dtype=float32)}}}

Thus each leaf in the tree structure of `jax.hessian(fun)(x)` corresponds to a leaf of `fun(x)` and a pair of leaves of `x`. For each leaf in `jax.hessian(fun)(x)`, if the corresponding array leaf of `fun(x)` has shape `(out_1,`` ``out_2,`` ``...)` and the corresponding array leaves of `x` have shape `(in_1_1,`` ``in_1_2,`` ``...)` and `(in_2_1,`` ``in_2_2,`` ``...)` respectively, then the Hessian leaf has shape `(out_1,`` ``out_2,`` ``...,`` ``in_1_1,`` ``in_1_2,`` ``...,`` ``in_2_1,`` ``in_2_2,`` ``...)`. In other words, the Python tree structure represents the block structure of the Hessian, with blocks determined by the input and output pytrees.

In particular, an array is produced (with no pytrees involved) when the function input `x` and output `fun(x)` are each a single array, as in the `g` example above. If `fun(x)` has shape `(out1,`` ``out2,`` ``...)` and `x` has shape `(in1,`` ``in2,`` ``...)` then `jax.hessian(fun)(x)` has shape `(out1,`` ``out2,`` ``...,`` ``in1,`` ``in2,`` ``...,`` ``in1,`` ``in2,`` ``...)`. To flatten pytrees into 1D vectors, consider using `jax.flatten_util.flatten_pytree()`.

[](jax.jacrev.html "previous page")

previous

jax.jacrev

[](jax.jvp.html "next page")

next

jax.jvp

Contents

- [`hessian()`](#jax.hessian)

By The JAX authors

© Copyright 2024, The JAX Authors.\
