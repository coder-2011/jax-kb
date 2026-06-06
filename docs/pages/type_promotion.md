- [](index.html)
- [Notes](notes.html)
- Type promotion semantics

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/type_promotion.rst "Download source file")
-  .pdf

# Type promotion semantics

## Contents

- [Effects of Python operator dispatch](#effects-of-python-operator-dispatch)
- [Weakly-typed values in JAX](#weakly-typed-values-in-jax)
- [Strict dtype promotion](#strict-dtype-promotion)

# Type promotion semantics[\#](#type-promotion-semantics "Link to this heading")

This document describes JAX’s type promotion rules–i.e., the result of [`jax.numpy.promote_types()`](_autosummary/jax.numpy.promote_types.html#jax.numpy.promote_types "jax.numpy.promote_types") for each pair of types. For some background on the considerations that went into the design of what is described below, see [Design of Type Promotion Semantics for JAX](https://docs.jax.dev/en/latest/jep/9407-type-promotion.html).

JAX’s type promotion behavior is determined via the following type promotion lattice:

![](_images/type_lattice.svg)

where, for example:

- `b1` means `np.bool_`,

- `i2` means `np.int16`,

- `u4` means `np.uint32`,

- `bf` means `np.bfloat16`,

- `f2` means `np.float16`,

- `c8` means `np.complex64`,

- `i*` means Python `int` or weakly-typed `int`,

- `f*` means Python `float` or weakly-typed `float`, and

- `c*` means Python `complex` or weakly-typed `complex`.

(for more about weak types, see [Weakly-typed values in JAX](#weak-types) below).

Promotion between any two types is given by their [join](https://en.wikipedia.org/wiki/Join_and_meet) on this lattice, which generates the following binary promotion table:

|     | b1  | u1  | u2  | u4  | u8  | i1  | i2  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | i\* | f\* | c\* |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| b1  | b1  | u1  | u2  | u4  | u8  | i1  | i2  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | i\* | f\* | c\* |
| u1  | u1  | u1  | u2  | u4  | u8  | i2  | i2  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | u1  | f\* | c\* |
| u2  | u2  | u2  | u2  | u4  | u8  | i4  | i4  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | u2  | f\* | c\* |
| u4  | u4  | u4  | u4  | u4  | u8  | i8  | i8  | i8  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | u4  | f\* | c\* |
| u8  | u8  | u8  | u8  | u8  | u8  | f\* | f\* | f\* | f\* | bf  | f2  | f4  | f8  | c8  | c16 | u8  | f\* | c\* |
| i1  | i1  | i2  | i4  | i8  | f\* | i1  | i2  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | i1  | f\* | c\* |
| i2  | i2  | i2  | i4  | i8  | f\* | i2  | i2  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | i2  | f\* | c\* |
| i4  | i4  | i4  | i4  | i8  | f\* | i4  | i4  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | i4  | f\* | c\* |
| i8  | i8  | i8  | i8  | i8  | f\* | i8  | i8  | i8  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | i8  | f\* | c\* |
| bf  | bf  | bf  | bf  | bf  | bf  | bf  | bf  | bf  | bf  | bf  | f4  | f4  | f8  | c8  | c16 | bf  | bf  | c8  |
| f2  | f2  | f2  | f2  | f2  | f2  | f2  | f2  | f2  | f2  | f4  | f2  | f4  | f8  | c8  | c16 | f2  | f2  | c8  |
| f4  | f4  | f4  | f4  | f4  | f4  | f4  | f4  | f4  | f4  | f4  | f4  | f4  | f8  | c8  | c16 | f4  | f4  | c8  |
| f8  | f8  | f8  | f8  | f8  | f8  | f8  | f8  | f8  | f8  | f8  | f8  | f8  | f8  | c16 | c16 | f8  | f8  | c16 |
| c8  | c8  | c8  | c8  | c8  | c8  | c8  | c8  | c8  | c8  | c8  | c8  | c8  | c16 | c8  | c16 | c8  | c8  | c8  |
| c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 | c16 |
| i\* | i\* | u1  | u2  | u4  | u8  | i1  | i2  | i4  | i8  | bf  | f2  | f4  | f8  | c8  | c16 | i\* | f\* | c\* |
| f\* | f\* | f\* | f\* | f\* | f\* | f\* | f\* | f\* | f\* | bf  | f2  | f4  | f8  | c8  | c16 | f\* | f\* | c\* |
| c\* | c\* | c\* | c\* | c\* | c\* | c\* | c\* | c\* | c\* | c8  | c8  | c8  | c16 | c8  | c16 | c\* | c\* | c\* |

Jax’s type promotion rules differ from those of NumPy, as given by [`numpy.promote_types()`](https://numpy.org/doc/stable/reference/generated/numpy.promote_types.html#numpy.promote_types "(in NumPy v2.4)"), in those cells highlighted with a green background in the table above. There are three key classes of differences:

- When promoting a weakly typed value against a typed JAX value of the same category, JAX always prefers the precision of the JAX value. For example, `jnp.int16(1)`` ``+`` ``1` will return `int16` rather than promoting to `int64` as in NumPy. Note that this applies only to Python scalar values; if the constant is a NumPy array then the above lattice is used for type promotion. For example, `jnp.int16(1)`` ``+`` ``np.array(1)` will return `int64`.

- When promoting an integer or boolean type against a floating-point or complex type, JAX always prefers the type of the floating-point or complex type.

- JAX supports the [bfloat16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) non-standard 16-bit floating point type (`jax.numpy.bfloat16`), which is useful for neural network training. The only notable promotion behavior is with respect to IEEE-754 `float16`, with which `bfloat16` promotes to a `float32`.

The differences between NumPy and JAX are motivated by the fact that accelerator devices, such as GPUs and TPUs, either pay a significant performance penalty to use 64-bit floating point types (GPUs) or do not support 64-bit floating point types at all (TPUs). Classic NumPy’s promotion rules are too willing to overpromote to 64-bit types, which is problematic for a system designed to run on accelerators.

JAX uses floating point promotion rules that are more suited to modern accelerator devices and are less aggressive about promoting floating point types. The promotion rules used by JAX for floating-point types are similar to those used by PyTorch.

## Effects of Python operator dispatch[\#](#effects-of-python-operator-dispatch "Link to this heading")

Keep in mind that Python operators like + will dispatch based on the Python type of the two values being added. This means that, for example, `np.int16(1)`` ``+`` ``1` will promote using NumPy rules, whereas `jnp.int16(1)`` ``+`` ``1` will promote using JAX rules. This can lead to potentially confusing non-associative promotion semantics when the two types of promotion are combined; for example with `np.int16(1)`` ``+`` ``1`` ``+`` ``jnp.int16(1)`.

## Weakly-typed values in JAX[\#](#weakly-typed-values-in-jax "Link to this heading")

*Weakly-typed* values in JAX can in most cases be thought of as having promotion behavior equivalent to that of Python scalars, such as the integer scalar `2` in the following:

    >>> x = jnp.arange(5, dtype='int8')
    >>> 2 * x
    Array([0, 2, 4, 6, 8], dtype=int8)

JAX’s weak type framework is designed to prevent unwanted type promotion within binary operations between JAX values and values with no explicitly user-specified type, such as Python scalar literals. For example, if `2` were not treated as weakly-typed, the expression above would lead to an implicit type promotion:

    >>> jnp.int32(2) * x
    Array([0, 2, 4, 6, 8], dtype=int32)

When used in JAX, Python scalars are sometimes promoted to `DeviceArray` objects, for example during JIT compilation. To maintain the desired promotion semantics in this case, `DeviceArray` objects carry a `weak_type` flag that can be seen in an array’s string representation:

    >>> jnp.asarray(2)
    Array(2, dtype=int32, weak_type=True)

If the `dtype` is specified explicitly, it will instead result in a standard strongly-typed array value:

    >>> jnp.asarray(2, dtype='int32')
    Array(2, dtype=int32)

## Strict dtype promotion[\#](#strict-dtype-promotion "Link to this heading")

In some contexts it can be useful to disable implicit type promotion behavior, and instead require all promotions to be explicit. This can be done in JAX by setting the `jax_numpy_dtype_promotion` flag to `'strict'`. Locally, it can be done with acontext manager:

    >>> x = jnp.float32(1)
    >>> y = jnp.int32(1)
    >>> with jax.numpy_dtype_promotion('strict'):
    ...   z = x + y  
    ...
    Traceback (most recent call last):
    TypePromotionError: Input dtypes ('float32', 'int32') have no available implicit
    dtype promotion path when jax_numpy_dtype_promotion=strict. Try explicitly casting
    inputs to the desired output type, or set jax_numpy_dtype_promotion=standard.

For convenience, strict promotion mode will still allow safe weakly-typed promotions, so you can still write code code that mixes JAX arrays and Python scalars:

    >>> with jax.numpy_dtype_promotion('strict'):
    ...   z = x + 1
    >>> print(z)
    2.0

If you would prefer to set the configuration globally, you can do so using the standard configuration update:

    jax.config.update('jax_numpy_dtype_promotion', 'strict')

To restore the default standard type promotion, set this configuration to `'standard'`:

    jax.config.update('jax_numpy_dtype_promotion', 'standard')

[](rank_promotion_warning.html "previous page")

previous

Rank promotion warning

[](default_dtypes.html "next page")

next

Default dtypes and the X64 flag

Contents

- [Effects of Python operator dispatch](#effects-of-python-operator-dispatch)
- [Weakly-typed values in JAX](#weakly-typed-values-in-jax)
- [Strict dtype promotion](#strict-dtype-promotion)

By The JAX authors

© Copyright 2024, The JAX Authors.\
