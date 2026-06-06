- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.fromstring

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.fromstring.rst "Download source file")
-  .pdf

# jax.numpy.fromstring

## Contents

- [`fromstring()`](#jax.numpy.fromstring)

# jax.numpy.fromstring[\#](#jax-numpy-fromstring "Link to this heading")

jax.numpy.fromstring(*string*, *dtype=\<class 'float'\>*, *count=-1*, *\**, *sep*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L5688-L5715)[\#](#jax.numpy.fromstring "Link to this definition")  
Convert a string of text into 1-D JAX array.

JAX implementation of [`numpy.fromstring()`](https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html#numpy.fromstring "(in NumPy v2.4)").

Parameters:  
- **string** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – input string containing the data.

- **dtype** (*DTypeLike*) – optional. Desired data type for the array. Default is `float`.

- **count** ([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")) – optional integer specifying the number of items to read from the string. If -1 (default), all items are read.

- **sep** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)")) – the string used to separate values in the input string.

Returns:  
A 1-D JAX array containing the parsed data from the input string.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.frombuffer()`](jax.numpy.frombuffer.html#jax.numpy.frombuffer "jax.numpy.frombuffer"): construct a JAX array from an object that implements the buffer interface.

Examples

    >>> jnp.fromstring("1 2 3", dtype=int, sep=" ")
    Array([1, 2, 3], dtype=int32)
    >>> jnp.fromstring("0.1, 0.2, 0.3", dtype=float, count=2, sep=",")
    Array([0.1, 0.2], dtype=float32)

[](jax.numpy.frompyfunc.html "previous page")

previous

jax.numpy.frompyfunc

[](jax.numpy.from_dlpack.html "next page")

next

jax.numpy.from_dlpack

Contents

- [`fromstring()`](#jax.numpy.fromstring)

By The JAX authors

© Copyright 2024, The JAX Authors.\
