- [](../index.html)
- [API Reference](../jax.html)
- [`jax.numpy` module](../jax.numpy.html)
- jax.numpy.generic

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.numpy.generic.rst "Download source file")
-  .pdf

# jax.numpy.generic

## Contents

- [`generic`](#jax.numpy.generic)
  - [`generic.__init__()`](#jax.numpy.generic.__init__)

# jax.numpy.generic[\#](#jax-numpy-generic "Link to this heading")

*class* jax.numpy.generic[\#](#jax.numpy.generic "Link to this definition")  
Base class for numpy scalar types.

Class from which most (all?) numpy scalar types are derived. For consistency, exposes the same API as ndarray, despite many consequent attributes being either “get-only,” or completely irrelevant. This is the class from which it is strongly suggested users should derive custom scalar types.

\_\_init\_\_()[\#](#jax.numpy.generic.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.numpy.generic.__init__ "jax.numpy.generic.__init__")() |  |
| `all`(\[axis, out, keepdims, where\]) | Scalar method identical to ndarray.all. |
| `any`(\[axis, out, keepdims, where\]) | Scalar method identical to ndarray.any. |
| `argmax`(\[axis, out, keepdims\]) | Scalar method identical to ndarray.argmax. |
| `argmin`(\[axis, out, keepdims\]) | Scalar method identical to ndarray.argmin. |
| `argsort`(\[axis, kind, order, stable\]) | Scalar method identical to ndarray.argsort. |
| `astype`(dtype\[, order, casting, subok, copy\]) | Scalar method identical to ndarray.astype. |
| `byteswap`(\[inplace\]) | Scalar method identical to ndarray.byteswap. |
| `choose`(choices\[, out, mode\]) | Scalar method identical to ndarray.choose. |
| `clip`(\[min, max, out\]) | Scalar method identical to ndarray.clip. |
| `compress`(condition\[, axis, out\]) | Scalar method identical to ndarray.compress. |
| `conj`() | Scalar method identical to ndarray.conj. |
| `conjugate`() | Scalar method identical to ndarray.conjugate. |
| `copy`(\[order\]) | Scalar method identical to ndarray.copy. |
| `cumprod`(\[axis, dtype, out\]) | Scalar method identical to ndarray.cumprod. |
| `cumsum`(\[axis, dtype, out\]) | Scalar method identical to ndarray.cumsum. |
| `diagonal`(\[offset, axis1, axis2\]) | Scalar method identical to ndarray.diagonal. |
| `dump`(file) | Scalar method identical to ndarray.dump. |
| `dumps`() | Scalar method identical to ndarray.dumps. |
| `fill`(value) | Scalar method identical to ndarray.fill. |
| `flatten`(\[order\]) | Scalar method identical to ndarray.flatten. |
| `getfield`(dtype\[, offset\]) | Scalar method identical to ndarray.getfield. |
| `item`(\*args) | Scalar method identical to ndarray.item. |
| `max`(\[axis, out\]) | Scalar method identical to ndarray.max. |
| `mean`(\[axis, dtype, out\]) | Scalar method identical to ndarray.mean. |
| `min`(\[axis, out\]) | Scalar method identical to ndarray.min. |
| `nonzero`() | Scalar method identical to ndarray.nonzero. |
| `prod`(\[axis, dtype, out\]) | Scalar method identical to ndarray.prod. |
| `put`(indices, values, /\[, mode\]) | Scalar method identical to ndarray.put. |
| `ravel`(\[order\]) | Scalar method identical to ndarray.ravel. |
| `repeat`(repeats, /\[, axis\]) | Scalar method identical to ndarray.repeat. |
| `reshape`(\*shape\[, order, copy\]) | Scalar method identical to ndarray.reshape. |
| `resize`(\*new_shape\[, refcheck\]) | Scalar method identical to ndarray.resize. |
| `round`(\[decimals, out\]) | Scalar method identical to ndarray.round. |
| `searchsorted`(v, /\[, side, sorter\]) | Scalar method identical to ndarray.searchsorted. |
| `setfield`(val, /, dtype\[, offset\]) | Scalar method identical to ndarray.setfield. |
| `setflags`(\*\[, write, align, uic\]) | Scalar method identical to ndarray.setflags. |
| `sort`(\[axis, kind, order, stable\]) | Scalar method identical to ndarray.sort. |
| `squeeze`(\[axis\]) | Scalar method identical to ndarray.squeeze. |
| `std`(\[axis, dtype, out, ddof\]) | Scalar method identical to ndarray.std. |
| `sum`(\[axis, dtype, out\]) | Scalar method identical to ndarray.sum. |
| `swapaxes`(axis1, axis2, /) | Scalar method identical to ndarray.swapaxes. |
| `take`(indices, /\[, axis, out, mode\]) | Scalar method identical to ndarray.take. |
| `to_device`(device, /, \*\[, stream\]) | Scalar method identical to ndarray.to_device. |
| `tobytes`(\[order\]) | Scalar method identical to ndarray.tobytes. |
| `tofile`(fid, /\[, sep, format\]) | Scalar method identical to ndarray.tofile. |
| `tolist`() | Scalar method identical to ndarray.tolist. |
| `trace`(\[offset, axis1, axis2, dtype, out\]) | Scalar method identical to ndarray.trace. |
| `transpose`(\*axes) | Scalar method identical to ndarray.transpose. |
| `var`(\[axis, dtype, out, ddof\]) | Scalar method identical to ndarray.var. |
| `view`(\*args, \*\*kwargs) | Scalar method identical to ndarray.view. |

Attributes

|            |                                             |
|------------|---------------------------------------------|
| `T`        | Scalar attribute identical to ndarray.T.    |
| `base`     | Scalar attribute identical to ndarray.base. |
| `data`     | Pointer to start of data.                   |
| `device`   |                                             |
| `dtype`    | Get array data-descriptor.                  |
| `flags`    | The integer value of flags.                 |
| `flat`     | A 1-D view of the scalar.                   |
| `imag`     | The imaginary part of the scalar.           |
| `itemsize` | The length of one element in bytes.         |
| `nbytes`   |                                             |
| `ndim`     | The number of array dimensions.             |
| `real`     | The real part of the scalar.                |
| `shape`    | Tuple of array dimensions.                  |
| `size`     | The number of elements in the gentype.      |
| `strides`  | Tuple of bytes steps in each dimension.     |

[](jax.numpy.gcd.html "previous page")

previous

jax.numpy.gcd

[](jax.numpy.geomspace.html "next page")

next

jax.numpy.geomspace

Contents

- [`generic`](#jax.numpy.generic)
  - [`generic.__init__()`](#jax.numpy.generic.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
