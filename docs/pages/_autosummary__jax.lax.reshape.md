- [](../index.html)
- [API Reference](../jax.html)
- [`jax.lax` module](../jax.lax.html)
- jax.lax.reshape

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.lax.reshape.rst "Download source file")
-  .pdf

# jax.lax.reshape

## Contents

- [`reshape()`](#jax.lax.reshape)

# jax.lax.reshape[\#](#jax-lax-reshape "Link to this heading")

jax.lax.reshape(*operand*, *new_sizes*, *dimensions=None*, *\**, *out_sharding=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/lax/lax.py#L2972-L3033)[\#](#jax.lax.reshape "Link to this definition")  
Wraps XLA’s [Reshape](https://www.openxla.org/xla/operation_semantics#reshape) operator.

For inserting/removing dimensions of size 1, prefer using `lax.squeeze` / `lax.expand_dims`. These preserve information about axis identity that may be useful for advanced transformation rules.

Parameters:  
- **operand** (*ArrayLike*) – array to be reshaped.

- **new_sizes** (*Shape*) – sequence of integers specifying the resulting shape. The size of the final array must match the size of the input.

- **dimensions** (*Sequence\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\]* *\|* *None*) – optional sequence of integers specifying the permutation order of the input shape. If specified, the length must match `operand.shape`.

- **out_sharding** ([*NamedSharding*](../jax.sharding.html#jax.sharding.NamedSharding "jax.sharding.NamedSharding") *\|* *P* *\|* *None*)

Returns:  
reshaped array.

Return type:  
out

Examples

Simple reshaping from one to two dimensions:

    >>> x = jnp.arange(6)
    >>> y = reshape(x, (2, 3))
    >>> y
    Array([[0, 1, 2],
                 [3, 4, 5]], dtype=int32)

Reshaping back to one dimension:

    >>> reshape(y, (6,))
    Array([0, 1, 2, 3, 4, 5], dtype=int32)

Reshaping to one dimension with permutation of dimensions:

    >>> reshape(y, (6,), (1, 0))
    Array([0, 3, 1, 4, 2, 5], dtype=int32)

[](jax.lax.rem.html "previous page")

previous

jax.lax.rem

[](jax.lax.rev.html "next page")

next

jax.lax.rev

Contents

- [`reshape()`](#jax.lax.reshape)

By The JAX authors

© Copyright 2024, The JAX Authors.\
