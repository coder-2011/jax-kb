- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.finfo

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.finfo.rst "Download source file")
-  .pdf

# jax.numpy.finfo

## Contents

- [`finfo`](#jax.numpy.finfo)
  - [`finfo.bits`](#jax.numpy.finfo.bits)
  - [`finfo.dtype`](#jax.numpy.finfo.dtype)
  - [`finfo.eps`](#jax.numpy.finfo.eps)
  - [`finfo.epsneg`](#jax.numpy.finfo.epsneg)
  - [`finfo.iexp`](#jax.numpy.finfo.iexp)
  - [`finfo.machep`](#jax.numpy.finfo.machep)
  - [`finfo.max`](#jax.numpy.finfo.max)
  - [`finfo.maxexp`](#jax.numpy.finfo.maxexp)
  - [`finfo.min`](#jax.numpy.finfo.min)
  - [`finfo.minexp`](#jax.numpy.finfo.minexp)
  - [`finfo.negep`](#jax.numpy.finfo.negep)
  - [`finfo.nexp`](#jax.numpy.finfo.nexp)
  - [`finfo.nmant`](#jax.numpy.finfo.nmant)
  - [`finfo.precision`](#jax.numpy.finfo.precision)
  - [`finfo.resolution`](#jax.numpy.finfo.resolution)
  - [`finfo.tiny`](#jax.numpy.finfo.tiny)
  - [`finfo.smallest_normal`](#jax.numpy.finfo.smallest_normal)
  - [`finfo.smallest_subnormal`](#jax.numpy.finfo.smallest_subnormal)
  - [`finfo.__init__()`](#jax.numpy.finfo.__init__)

# jax.numpy.finfo[\#](#jax-numpy-finfo "Link to this heading")

*class* jax.numpy.finfo(*dtype*)[\#](#jax.numpy.finfo "Link to this definition")  
Machine limits for floating point types.

bits[\#](#jax.numpy.finfo.bits "Link to this definition")  
The number of bits occupied by the type.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

dtype[\#](#jax.numpy.finfo.dtype "Link to this definition")  
Returns the dtype for which finfo returns information. For complex input, the returned dtype is the associated `float*` dtype for its real and complex components.

Type:  
[dtype](jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype")

eps[\#](#jax.numpy.finfo.eps "Link to this definition")  
The difference between 1.0 and the next smallest representable float larger than 1.0. For example, for 64-bit binary floats in the IEEE-754 standard, `eps`` ``=`` ``2**-52`, approximately 2.22e-16.

Type:  
[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

epsneg[\#](#jax.numpy.finfo.epsneg "Link to this definition")  
The difference between 1.0 and the next smallest representable float less than 1.0. For example, for 64-bit binary floats in the IEEE-754 standard, `epsneg`` ``=`` ``2**-53`, approximately 1.11e-16.

Type:  
[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

iexp[\#](#jax.numpy.finfo.iexp "Link to this definition")  
The number of bits in the exponent portion of the floating point representation.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

machep[\#](#jax.numpy.finfo.machep "Link to this definition")  
The exponent that yields eps.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

max[\#](#jax.numpy.finfo.max "Link to this definition")  
The largest representable number.

Type:  
floating point number of the appropriate type

maxexp[\#](#jax.numpy.finfo.maxexp "Link to this definition")  
The smallest positive power of the base (2) that causes overflow. Corresponds to the C standard MAX_EXP.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

min[\#](#jax.numpy.finfo.min "Link to this definition")  
The smallest representable number, typically `-max`.

Type:  
floating point number of the appropriate type

minexp[\#](#jax.numpy.finfo.minexp "Link to this definition")  
The most negative power of the base (2) consistent with there being no leading 0’s in the mantissa. Corresponds to the C standard MIN_EXP - 1.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

negep[\#](#jax.numpy.finfo.negep "Link to this definition")  
The exponent that yields epsneg.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

nexp[\#](#jax.numpy.finfo.nexp "Link to this definition")  
The number of bits in the exponent including its sign and bias.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

nmant[\#](#jax.numpy.finfo.nmant "Link to this definition")  
The number of explicit bits in the mantissa (excluding the implicit leading bit for normalized numbers).

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

precision[\#](#jax.numpy.finfo.precision "Link to this definition")  
The approximate number of decimal digits to which this kind of float is precise.

Type:  
[int](https://docs.python.org/3/library/functions.html#int "(in Python v3.14)")

resolution[\#](#jax.numpy.finfo.resolution "Link to this definition")  
The approximate decimal resolution of this type, i.e., `10**-precision`.

Type:  
floating point number of the appropriate type

tiny[\#](#jax.numpy.finfo.tiny "Link to this definition")  
An alias for smallest_normal, kept for backwards compatibility.

Type:  
[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

smallest_normal[\#](#jax.numpy.finfo.smallest_normal "Link to this definition")  
The smallest positive floating point number with 1 as leading bit in the mantissa following IEEE-754 (see Notes).

Type:  
[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

smallest_subnormal[\#](#jax.numpy.finfo.smallest_subnormal "Link to this definition")  
The smallest positive floating point number with 0 as leading bit in the mantissa following IEEE-754.

Type:  
[float](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")

Parameters:  
**dtype** ([*float*](https://docs.python.org/3/library/functions.html#float "(in Python v3.14)")*,* [*dtype*](jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype")*, or* *instance*) – Kind of floating point or complex floating point data-type about which to get information.

See also

[`iinfo`](jax.numpy.iinfo.html#jax.numpy.iinfo "jax.numpy.iinfo")  
The equivalent for integer data types.

[`spacing`](jax.numpy.spacing.html#jax.numpy.spacing "jax.numpy.spacing")  
The distance between a value and the nearest adjacent number

[`nextafter`](jax.numpy.nextafter.html#jax.numpy.nextafter "jax.numpy.nextafter")  
The next floating point value after x1 towards x2

Notes

For developers of NumPy: do not instantiate this at the module level. The initial calculation of these parameters is expensive and negatively impacts import times. These objects are cached, so calling `finfo()` repeatedly inside your functions is not a problem.

Note that `smallest_normal` is not actually the smallest positive representable value in a NumPy floating point type. As in the IEEE-754 standard [^1], NumPy floating point types make use of subnormal numbers to fill the gap between 0 and `smallest_normal`. However, subnormal numbers may have significantly reduced precision [^2].

For `longdouble`, the representation varies across platforms. On most platforms it is IEEE 754 binary128 (quad precision) or binary64-extended (80-bit extended precision). On PowerPC systems, it may use the IBM double-double format (a pair of float64 values), which has special characteristics for precision and range.

This function can also be used for complex data types as well. If used, the output will be the same as the corresponding real float type (e.g. numpy.finfo(numpy.csingle) is the same as numpy.finfo(numpy.single)). However, the output is true for the real and imaginary components.

References

\[[1](#id1)\]

IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2008, pp.1-70, 2008, [https://doi.org/10.1109/IEEESTD.2008.4610935](https://doi.org/10.1109/IEEESTD.2008.4610935)

\[[2](#id2)\]

Wikipedia, “Denormal Numbers”, [https://en.wikipedia.org/wiki/Denormal_number](https://en.wikipedia.org/wiki/Denormal_number)

Examples

    >>> import numpy as np
    >>> np.finfo(np.float64).dtype
    dtype('float64')
    >>> np.finfo(np.complex64).dtype
    dtype('float32')

\_\_init\_\_()[\#](#jax.numpy.finfo.__init__ "Link to this definition")  

Methods

|                                                                      |     |
|----------------------------------------------------------------------|-----|
| [`__init__`](#jax.numpy.finfo.__init__ "jax.numpy.finfo.__init__")() |     |

Attributes

|  |  |
|----|----|
| [`epsneg`](#jax.numpy.finfo.epsneg "jax.numpy.finfo.epsneg") |  |
| [`iexp`](#jax.numpy.finfo.iexp "jax.numpy.finfo.iexp") |  |
| [`machep`](#jax.numpy.finfo.machep "jax.numpy.finfo.machep") |  |
| [`negep`](#jax.numpy.finfo.negep "jax.numpy.finfo.negep") |  |
| [`nexp`](#jax.numpy.finfo.nexp "jax.numpy.finfo.nexp") |  |
| [`resolution`](#jax.numpy.finfo.resolution "jax.numpy.finfo.resolution") |  |
| [`tiny`](#jax.numpy.finfo.tiny "jax.numpy.finfo.tiny") | Return the value for tiny, alias of smallest_normal. |

[](jax.numpy.fill_diagonal.html "previous page")

previous

jax.numpy.fill_diagonal

[](jax.numpy.flatnonzero.html "next page")

next

jax.numpy.flatnonzero

Contents

- [`finfo`](#jax.numpy.finfo)
  - [`finfo.bits`](#jax.numpy.finfo.bits)
  - [`finfo.dtype`](#jax.numpy.finfo.dtype)
  - [`finfo.eps`](#jax.numpy.finfo.eps)
  - [`finfo.epsneg`](#jax.numpy.finfo.epsneg)
  - [`finfo.iexp`](#jax.numpy.finfo.iexp)
  - [`finfo.machep`](#jax.numpy.finfo.machep)
  - [`finfo.max`](#jax.numpy.finfo.max)
  - [`finfo.maxexp`](#jax.numpy.finfo.maxexp)
  - [`finfo.min`](#jax.numpy.finfo.min)
  - [`finfo.minexp`](#jax.numpy.finfo.minexp)
  - [`finfo.negep`](#jax.numpy.finfo.negep)
  - [`finfo.nexp`](#jax.numpy.finfo.nexp)
  - [`finfo.nmant`](#jax.numpy.finfo.nmant)
  - [`finfo.precision`](#jax.numpy.finfo.precision)
  - [`finfo.resolution`](#jax.numpy.finfo.resolution)
  - [`finfo.tiny`](#jax.numpy.finfo.tiny)
  - [`finfo.smallest_normal`](#jax.numpy.finfo.smallest_normal)
  - [`finfo.smallest_subnormal`](#jax.numpy.finfo.smallest_subnormal)
  - [`finfo.__init__()`](#jax.numpy.finfo.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\

[^1]:

[^2]:
