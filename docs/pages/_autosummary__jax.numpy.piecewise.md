- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.piecewise

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.piecewise.rst "Download source file")
-  .pdf

# jax.numpy.piecewise

## Contents

- [`piecewise()`](#jax.numpy.piecewise)

# jax.numpy.piecewise[\#](#jax-numpy-piecewise "Link to this heading")

jax.numpy.piecewise(*x*, *condlist*, *funclist*, *\*args*, *\*\*kw*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L9458-L9541)[\#](#jax.numpy.piecewise "Link to this definition")  
Evaluate a function defined piecewise across the domain.

JAX implementation of [`numpy.piecewise()`](https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html#numpy.piecewise "(in NumPy v2.4)"), in terms of [`jax.lax.switch()`](jax.lax.switch.html#jax.lax.switch "jax.lax.switch").

Note

Unlike [`numpy.piecewise()`](https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html#numpy.piecewise "(in NumPy v2.4)"), [`jax.numpy.piecewise()`](#jax.numpy.piecewise "jax.numpy.piecewise") requires functions in `funclist` to be traceable by JAX, as it is implemented via [`jax.lax.switch()`](jax.lax.switch.html#jax.lax.switch "jax.lax.switch").

Parameters:  
- **x** (*ArrayLike*) – array of input values.

- **condlist** ([*Array*](jax.Array.html#jax.Array "jax.Array") *\|* *Sequence\[ArrayLike\]*) – boolean array or sequence of boolean arrays corresponding to the functions in `funclist`. If a sequence of arrays, the length of each array must match the length of `x`

- **funclist** ([*list*](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")*\[ArrayLike* *\|* [*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "jax.extend.linear_util.Callable")*\[...,* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]\]*) – list of arrays or functions; must either be the same length as `condlist`, or have length `len(condlist)`` ``+`` ``1`, in which case the last entry is the default applied when none of the conditions are True. Alternatively, entries of `funclist` may be numerical values, in which case they indicate a constant function.

- **args** – additional arguments are passed to each function in `funclist`.

- **kwargs** – additional arguments are passed to each function in `funclist`.

Returns:  
An array which is the result of evaluating the functions on `x` at the specified conditions.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.lax.switch()`](jax.lax.switch.html#jax.lax.switch "jax.lax.switch"): choose between *N* functions based on an index.

- [`jax.lax.cond()`](jax.lax.cond.html#jax.lax.cond "jax.lax.cond"): choose between two functions based on a boolean condition.

- [`jax.numpy.where()`](jax.numpy.where.html#jax.numpy.where "jax.numpy.where"): choose between two results based on a boolean mask.

- [`jax.lax.select()`](jax.lax.select.html#jax.lax.select "jax.lax.select"): choose between two results based on a boolean mask.

- [`jax.lax.select_n()`](jax.lax.select_n.html#jax.lax.select_n "jax.lax.select_n"): choose between *N* results based on a boolean mask.

Examples

Here’s an example of a function which is zero for negative values, and linear for positive values:

    >>> x = jnp.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    >>> condlist = [x < 0, x >= 0]
    >>> funclist = [lambda x: 0 * x, lambda x: x]
    >>> jnp.piecewise(x, condlist, funclist)
    Array([0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=int32)

`funclist` can also contain a simple scalar value for constant functions:

    >>> condlist = [x < 0, x >= 0]
    >>> funclist = [0, lambda x: x]
    >>> jnp.piecewise(x, condlist, funclist)
    Array([0, 0, 0, 0, 0, 1, 2, 3, 4], dtype=int32)

You can specify a default value by appending an extra condition to `funclist`:

    >>> condlist = [x < -1, x > 1]
    >>> funclist = [lambda x: 1 + x, lambda x: x - 1, 0]
    >>> jnp.piecewise(x, condlist, funclist)
    Array([-3, -2,  -1,  0,  0,  0,  1,  2, 3], dtype=int32)

`condlist` may also be a simple array of scalar conditions, in which case the associated function applies to the whole range

    >>> condlist = jnp.array([False, True, False])
    >>> funclist = [lambda x: x * 0, lambda x: x * 10, lambda x: x * 100]
    >>> jnp.piecewise(x, condlist, funclist)
    Array([-40, -30, -20, -10,   0,  10,  20,  30,  40], dtype=int32)

[](jax.numpy.permute_dims.html "previous page")

previous

jax.numpy.permute_dims

[](jax.numpy.place.html "next page")

next

jax.numpy.place

Contents

- [`piecewise()`](#jax.numpy.piecewise)

By The JAX authors

© Copyright 2024, The JAX Authors.\
