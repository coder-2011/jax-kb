- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.dsplit

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.dsplit.rst "Download source file")
-  .pdf

# jax.numpy.dsplit

## Contents

- [`dsplit()`](#jax.numpy.dsplit)

# jax.numpy.dsplit[\#](#jax-numpy-dsplit "Link to this heading")

jax.numpy.dsplit(*ary*, *indices_or_sections*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3293-L3333)[\#](#jax.numpy.dsplit "Link to this definition")  
Split an array into sub-arrays depth-wise.

JAX implementation of [`numpy.dsplit()`](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html#numpy.dsplit "(in NumPy v2.4)").

Refer to the documentation of [`jax.numpy.split()`](jax.numpy.split.html#jax.numpy.split "jax.numpy.split") for details. `dsplit` is equivalent to `split` with `axis=2`.

Examples

    >>> x = jnp.arange(12).reshape(3, 1, 4)
    >>> print(x)
    [[[ 0  1  2  3]]

     [[ 4  5  6  7]]

     [[ 8  9 10 11]]]
    >>> x1, x2 = jnp.dsplit(x, 2)
    >>> print(x1)
    [[[0 1]]

     [[4 5]]

     [[8 9]]]
    >>> print(x2)
    [[[ 2  3]]

     [[ 6  7]]

     [[10 11]]]

See also

- [`jax.numpy.split()`](jax.numpy.split.html#jax.numpy.split "jax.numpy.split"): split an array along any axis.

- [`jax.numpy.vsplit()`](jax.numpy.vsplit.html#jax.numpy.vsplit "jax.numpy.vsplit"): split vertically, i.e. along axis=0

- [`jax.numpy.hsplit()`](jax.numpy.hsplit.html#jax.numpy.hsplit "jax.numpy.hsplit"): split horizontally, i.e. along axis=1

- [`jax.numpy.array_split()`](jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split"): like `split`, but allows `indices_or_sections` to be an integer that does not evenly divide the size of the array.

Parameters:  
- **ary** (*ArrayLike*)

- **indices_or_sections** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* *Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *ArrayLike*)

Return type:  
[list](https://docs.python.org/3/library/stdtypes.html#list "(in Python v3.14)")\[[Array](jax.Array.html#jax.Array "jax.Array")\]

[](jax.numpy.double.html "previous page")

previous

jax.numpy.double

[](jax.numpy.dstack.html "next page")

next

jax.numpy.dstack

Contents

- [`dsplit()`](#jax.numpy.dsplit)

By The JAX authors

© Copyright 2024, The JAX Authors.\
