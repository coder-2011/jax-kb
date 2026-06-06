- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.split

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.split.rst "Download source file")
-  .pdf

# jax.random.split

## Contents

- [`split()`](#jax.random.split)

# jax.random.split[\#](#jax-random-split "Link to this heading")

jax.random.split(*key*, *num=2*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L319-L332)[\#](#jax.random.split "Link to this definition")  
Splits a PRNG key into num new keys by adding a leading axis.

Parameters:  
- **key** (*ArrayLike*) – a PRNG key (from `key`, `split`, `fold_in`).

- **num** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)") *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*) – optional, a positive integer (or tuple of integers) indicating the number (or shape) of keys to produce. Defaults to 2.

Returns:  
An array-like object of num new PRNG keys.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

[](jax.random.fold_in.html "previous page")

previous

jax.random.fold_in

[](jax.random.clone.html "next page")

next

jax.random.clone

Contents

- [`split()`](#jax.random.split)

By The JAX authors

© Copyright 2024, The JAX Authors.\
