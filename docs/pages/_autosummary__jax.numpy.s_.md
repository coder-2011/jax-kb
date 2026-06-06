- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.s\_

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.s_.rst "Download source file")
-  .pdf

# jax.numpy.s\_

## Contents

- [`s_`](#jax.numpy.s_)

# jax.numpy.s\_[\#](#jax-numpy-s "Link to this heading")

jax.numpy.s\_ *= \<numpy.lib.\_index_tricks_impl.IndexExpression object\>*[\#](#jax.numpy.s_ "Link to this definition")  
A nicer way to build up index tuples for arrays.

Note

Use one of the two predefined instances `index_exp` or s\_ rather than directly using IndexExpression.

For any index combination, including slicing and axis insertion, `a[indices]` is the same as `a[np.index_exp[indices]]` for any array a. However, `np.index_exp[indices]` can be used anywhere in Python code and returns a tuple of slice objects that can be used in the construction of complex index expressions.

Parameters:  
**maketuple** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – If True, always returns a tuple.

See also

[`s_`](#jax.numpy.s_ "jax.numpy.s_")  
Predefined instance without tuple conversion: s\_ = IndexExpression(maketuple=False). The `index_exp` is another predefined instance that always returns a tuple: index_exp = IndexExpression(maketuple=True).

Notes

You can do all this with [`slice`](https://docs.python.org/3/library/functions.html#slice "(in Python v3.14)") plus a few special objects, but there’s a lot to remember and this version is simpler because it uses the standard array indexing syntax.

Examples

    >>> import numpy as np
    >>> np.s_[2::2]
    slice(2, None, 2)
    >>> np.index_exp[2::2]
    (slice(2, None, 2),)

    >>> np.array([0, 1, 2, 3, 4])[np.s_[2::2]]
    array([2, 4])

[](jax.numpy.round.html "previous page")

previous

jax.numpy.round

[](jax.numpy.save.html "next page")

next

jax.numpy.save

Contents

- [`s_`](#jax.numpy.s_)

By The JAX authors

© Copyright 2024, The JAX Authors.\
