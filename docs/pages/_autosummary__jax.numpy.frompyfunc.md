- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.frompyfunc

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.frompyfunc.rst "Download source file")
-  .pdf

# jax.numpy.frompyfunc

## Contents

- [`frompyfunc()`](#jax.numpy.frompyfunc)

# jax.numpy.frompyfunc[\#](#jax-numpy-frompyfunc "Link to this heading")

jax.numpy.frompyfunc(*func*, */*, *nin*, *nout*, *\**, *identity=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/ufunc_api.py#L598-L636)[\#](#jax.numpy.frompyfunc "Link to this definition")  
Create a JAX ufunc from an arbitrary JAX-compatible scalar function.

Parameters:  
- **func** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* *Any\]*) – a callable that takes nin scalar arguments and returns nout outputs.

- **nin** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer specifying the number of scalar inputs

- **nout** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – integer specifying the number of scalar outputs

- **identity** (*Any*) – (optional) a scalar specifying the identity of the operation, if any.

Returns:  
jax.numpy.ufunc wrapper of func.

Return type:  
wrapped

Examples

Here is an example of creating a ufunc similar to [`jax.numpy.add`](jax.numpy.add.html#jax.numpy.add "jax.numpy.add"):

    >>> import operator
    >>> add = frompyfunc(operator.add, nin=2, nout=1, identity=0)

Now all the standard [`jax.numpy.ufunc`](jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc") methods are available:

    >>> x = jnp.arange(4)
    >>> add(x, 10)
    Array([10, 11, 12, 13], dtype=int32)
    >>> add.outer(x, x)
    Array([[0, 1, 2, 3],
           [1, 2, 3, 4],
           [2, 3, 4, 5],
           [3, 4, 5, 6]], dtype=int32)
    >>> add.reduce(x)
    Array(6, dtype=int32)
    >>> add.accumulate(x)
    Array([0, 1, 3, 6], dtype=int32)
    >>> add.at(x, 1, 10, inplace=False)
    Array([ 0, 11,  2,  3], dtype=int32)

[](jax.numpy.fromiter.html "previous page")

previous

jax.numpy.fromiter

[](jax.numpy.fromstring.html "next page")

next

jax.numpy.fromstring

Contents

- [`frompyfunc()`](#jax.numpy.frompyfunc)

By The JAX authors

© Copyright 2024, The JAX Authors.\
