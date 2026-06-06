- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.can_cast

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.can_cast.rst "Download source file")
-  .pdf

# jax.numpy.can_cast

## Contents

- [`can_cast()`](#jax.numpy.can_cast)

# jax.numpy.can_cast[\#](#jax-numpy-can-cast "Link to this heading")

jax.numpy.can_cast(*from\_*, *to*, *casting='safe'*)[\#](#jax.numpy.can_cast "Link to this definition")  
Returns True if cast between data types can occur according to the casting rule.

Parameters:  
- **from** ([*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype")*,* *dtype specifier,* *NumPy scalar, or* *array*) – Data type, NumPy scalar, or array to cast from.

- **to** ([*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype") *or* *dtype specifier*) – Data type to cast to.

- **casting** (*{'no',* *'equiv',* *'safe',* *'same_kind',* *'unsafe'},* *optional*) –

  Controls what kind of data casting may occur.

  - ’no’ means the data types should not be cast at all.

  - ’equiv’ means only byte-order changes are allowed.

  - ’safe’ means only casts which can preserve values are allowed.

  - ’same_kind’ means only safe casts or casts within a kind, like float64 to float32, are allowed.

  - ’unsafe’ means any data conversions may be done.

Returns:  
**out** – True if cast can occur according to the casting rule.

Return type:  
[bool](https://docs.python.org/3/library/functions.html#bool "(in Python v3.14)")

Notes

Changed in version 2.0: This function does not support Python scalars anymore and does not apply any value-based logic for 0-D arrays and NumPy scalars.

See also

[`dtype`](jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype"), [`result_type`](jax.numpy.result_type.html#jax.numpy.result_type "jax.numpy.result_type")

Examples

Basic examples

    >>> import numpy as np
    >>> np.can_cast(np.int32, np.int64)
    True
    >>> np.can_cast(np.float64, complex)
    True
    >>> np.can_cast(complex, float)
    False

    >>> np.can_cast('i8', 'f8')
    True
    >>> np.can_cast('i8', 'f4')
    False
    >>> np.can_cast('i4', 'S4')
    False

[](jax.numpy.c_.html "previous page")

previous

jax.numpy.c\_

[](jax.numpy.cbrt.html "next page")

next

jax.numpy.cbrt

Contents

- [`can_cast()`](#jax.numpy.can_cast)

By The JAX authors

© Copyright 2024, The JAX Authors.\
