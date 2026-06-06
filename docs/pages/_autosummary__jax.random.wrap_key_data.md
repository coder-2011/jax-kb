- [](../index.html)
- [API Reference](../jax.html)
- [`jax.random` module](../jax.random.html)
- jax.random.wrap_key_data

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.random.wrap_key_data.rst "Download source file")
-  .pdf

# jax.random.wrap_key_data

## Contents

- [`wrap_key_data()`](#jax.random.wrap_key_data)

# jax.random.wrap_key_data[\#](#jax-random-wrap-key-data "Link to this heading")

jax.random.wrap_key_data(*key_bits_array*, *\**, *impl=None*, *dtype=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/random/core.py#L358-L397)[\#](#jax.random.wrap_key_data "Link to this definition")  
Wrap an array of key data bits into a PRNG key array.

Parameters:  
- **key_bits_array** ([*Array*](jax.Array.html#jax.Array "jax.Array")) – a `uint32` array with trailing shape corresponding to the key shape of the PRNG implementation specified by `impl`.

- **impl** (*PRNGSpecDesc* *\|* *None*) – optional, specifies a PRNG implementation, as in `random.key`.

- **dtype** (*KeyDTypeLike* *\|* *None*) – optional dtype or string name specifying the PRNG implementation (e.g. `jax.random.key_dtype('threefry2x32')` or `'threefry2x32'`).

Returns:  
A PRNG key array, whose dtype is a subdtype of `jax.dtypes.prng_key`  
corresponding to `impl`, and whose shape equals the leading shape of `key_bits_array.shape` up to the key bit dimensions.

Examples

Construct a key, and extract its data and dtype:

    >>> import jax
    >>> key = jax.random.key(42)
    >>> data = jax.random.key_data(key)
    >>> dtype = key.dtype

Reconstruct an equivalent key with [`wrap_key_data()`](#jax.random.wrap_key_data "jax.random.wrap_key_data"):

    >>> new_key = jax.random.wrap_key_data(data, dtype=dtype)
    >>> key == new_key
    Array(True, dtype=bool)

[](jax.random.key_data.html "previous page")

previous

jax.random.key_data

[](jax.random.fold_in.html "next page")

next

jax.random.fold_in

Contents

- [`wrap_key_data()`](#jax.random.wrap_key_data)

By The JAX authors

© Copyright 2024, The JAX Authors.\
