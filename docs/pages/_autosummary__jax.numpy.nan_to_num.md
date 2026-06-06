- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.nan_to_num

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.nan_to_num.rst "Download source file")
-  .pdf

# jax.numpy.nan_to_num

## Contents

- [`nan_to_num()`](#jax.numpy.nan_to_num)

# jax.numpy.nan_to_num[\#](#jax-numpy-nan-to-num "Link to this heading")

jax.numpy.nan_to_num(*x*, *copy=True*, *nan=0.0*, *posinf=None*, *neginf=None*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/lax_numpy.py#L3495-L3559)[\#](#jax.numpy.nan_to_num "Link to this definition")  
Replace NaN and infinite entries in an array.

JAX implementation of [`numpy.nan_to_num()`](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html#numpy.nan_to_num "(in NumPy v2.4)").

Parameters:  
- **x** (*ArrayLike*) – array of values to be replaced. If it does not have an inexact dtype it will be returned unmodified.

- **copy** ([*bool*](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")) – unused by JAX

- **nan** (*ArrayLike*) – value to substitute for NaN entries. Defaults to 0.0.

- **posinf** (*ArrayLike* *\|* *None*) – value to substitute for positive infinite entries. Defaults to the maximum representable value.

- **neginf** (*ArrayLike* *\|* *None*) – value to substitute for positive infinite entries. Defaults to the minimum representable value.

Returns:  
A copy of `x` with the requested substitutions.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array")

See also

- [`jax.numpy.isnan()`](jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan"): return True where the array contains NaN

- [`jax.numpy.isposinf()`](jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf"): return True where the array contains +inf

- [`jax.numpy.isneginf()`](jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf"): return True where the array contains -inf

Examples

    >>> x = jnp.array([0, jnp.nan, 1, jnp.inf, 2, -jnp.inf])

Default substitution values:

    >>> jnp.nan_to_num(x)
    Array([ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  3.4028235e+38,
            2.0000000e+00, -3.4028235e+38], dtype=float32)

Overriding substitutions for `-inf` and `+inf`:

    >>> jnp.nan_to_num(x, posinf=999, neginf=-999)
    Array([   0.,    0.,    1.,  999.,    2., -999.], dtype=float32)

If you only wish to substitute for NaN values while leaving `inf` values untouched, using [`where()`](jax.numpy.where.html#jax.numpy.where "jax.numpy.where") with [`jax.numpy.isnan()`](jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan") is a better option:

    >>> jnp.where(jnp.isnan(x), 0, x)
    Array([  0.,   0.,   1.,  inf,   2., -inf], dtype=float32)

[](jax.numpy.multiply.html "previous page")

previous

jax.numpy.multiply

[](jax.numpy.nanargmax.html "next page")

next

jax.numpy.nanargmax

Contents

- [`nan_to_num()`](#jax.numpy.nan_to_num)

By The JAX authors

© Copyright 2024, The JAX Authors.\
