- [](../index.html)
- [API Reference](../jax.html)
- [`jax.experimental` module](../jax.experimental.html)
- [`jax.experimental.pallas` module](../jax.experimental.pallas.html)
- jax.experimental.pallas.when

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.experimental.pallas.when.rst "Download source file")
-  .pdf

# jax.experimental.pallas.when

## Contents

- [`when()`](#jax.experimental.pallas.when)

# jax.experimental.pallas.when[\#](#jax-experimental-pallas-when "Link to this heading")

jax.experimental.pallas.when(*condition*, */*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/pallas/helpers.py#L68-L88)[\#](#jax.experimental.pallas.when "Link to this definition")  
Calls the decorated function when the condition is met.

Parameters:  
**condition** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)") *\|* [*Array*](jax.Array.html#jax.Array "jax.Array") *\|* [*ndarray*](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") *\|* [*bool*](https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.bool "(in NumPy v2.4)") *\|* [*number*](jax.numpy.number.html#jax.numpy.number "numpy.number") *\|* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)") *\|* [*complex*](https://docs.python.org/3/library/functions.html#complex "(in Python v3.14)")) – If a boolean, this is equivalent to `if`` ``condition:`` ``f()`. If an array, `when` produces a [`jax.lax.cond()`](jax.lax.cond.html#jax.lax.cond "jax.lax.cond") with the decorated function as the true branch.

Returns:  
A decorator.

Return type:  
[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")\[\[[*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")\[\[\], None\]\], None\]

[](jax.experimental.pallas.run_scoped.html "previous page")

previous

jax.experimental.pallas.run_scoped

[](jax.experimental.pallas.with_scoped.html "next page")

next

jax.experimental.pallas.with_scoped

Contents

- [`when()`](#jax.experimental.pallas.when)

By The JAX authors

© Copyright 2024, The JAX Authors.\
