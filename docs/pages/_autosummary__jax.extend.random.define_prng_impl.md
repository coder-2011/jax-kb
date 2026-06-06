- [](../index.html)
- [API Reference](../jax.html)
- [`jax.extend` module](../jax.extend.html)
- [`jax.extend.random` module](../jax.extend.random.html)
- jax.extend.random.define_prng_impl

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.extend.random.define_prng_impl.rst "Download source file")
-  .pdf

# jax.extend.random.define_prng_impl

## Contents

- [`define_prng_impl()`](#jax.extend.random.define_prng_impl)

# jax.extend.random.define_prng_impl[\#](#jax-extend-random-define-prng-impl "Link to this heading")

jax.extend.random.define_prng_impl(*\**, *key_shape*, *seed*, *split*, *random_bits*, *fold_in*, *name='\<unnamed\>'*, *tag='?'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/extend/random.py#L23-L34)[\#](#jax.extend.random.define_prng_impl "Link to this definition")  
Parameters:  
- **key_shape** ([*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]*)

- **seed** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **split** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **random_bits** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[*[*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*,* *...\]\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **fold_in** ([*Callable*](jax.extend.linear_util.Callable.html#jax.extend.linear_util.Callable "collections.abc.Callable")*\[\[*[*Array*](jax.Array.html#jax.Array "jax.Array")*,* [*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")*\],* [*Array*](jax.Array.html#jax.Array "jax.Array")*\]*)

- **name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

- **tag** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Return type:  
[*Hashable*](https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable "(in Python v3.14)")

[](../jax.extend.random.html "previous page")

previous

`jax.extend.random` module

[](jax.extend.random.seed_with_impl.html "next page")

next

jax.extend.random.seed_with_impl

Contents

- [`define_prng_impl()`](#jax.extend.random.define_prng_impl)

By The JAX authors

© Copyright 2024, The JAX Authors.\
