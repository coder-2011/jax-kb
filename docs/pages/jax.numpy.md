- [](index.html)
- [API Reference](jax.html)
- `jax.numpy` module

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](_sources/jax.numpy.rst "Download source file")
-  .pdf

# jax.numpy module

## Contents

- [jax.numpy.fft](#module-jax.numpy.fft)
- [jax.numpy.linalg](#module-jax.numpy.linalg)
- [JAX Array](#jax-array)
  - [Copying and Serialization](#copying-and-serialization)
- [Python Array API standard](#python-array-api-standard)

# `jax.numpy` module[\#](#module-jax.numpy "Link to this heading")

Implements the NumPy API, using the primitives in [`jax.lax`](jax.lax.html#module-jax.lax "jax.lax").

While JAX tries to follow the NumPy API as closely as possible, sometimes JAX cannot follow NumPy exactly.

- Notably, since JAX arrays are immutable, NumPy APIs that mutate arrays in-place cannot be implemented in JAX. However, often JAX is able to provide an alternative API that is purely functional. For example, instead of in-place array updates (`x[i]`` ``=`` ``y`), JAX provides an alternative pure indexed update function `x.at[i].set(y)` (see [`ndarray.at`](_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at")).

- Relatedly, some NumPy functions often return views of arrays when possible (examples are [`transpose()`](_autosummary/jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose") and [`reshape()`](_autosummary/jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape")). JAX versions of such functions will return copies instead, although such are often optimized away by XLA when sequences of operations are compiled using [`jax.jit()`](_autosummary/jax.jit.html#jax.jit "jax.jit").

- NumPy is very aggressive at promoting values to `float64` type. JAX sometimes is less aggressive about type promotion (See [Type promotion semantics](type_promotion.html#type-promotion)).

- Some NumPy routines have data-dependent output shapes (examples include [`unique()`](_autosummary/jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique") and [`nonzero()`](_autosummary/jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")). Because the XLA compiler requires array shapes to be known at compile time, such operations are not compatible with JIT. For this reason, JAX adds an optional `size` argument to such functions which may be specified statically in order to use them with JIT.

Nearly all applicable NumPy functions are implemented in the `jax.numpy` namespace; they are listed below.

|  |  |
|----|----|
| [`ndarray.at`](_autosummary/jax.numpy.ndarray.at.html#jax.numpy.ndarray.at "jax.numpy.ndarray.at") | Helper property for index update functionality. |
| [`abs`](_autosummary/jax.numpy.abs.html#jax.numpy.abs "jax.numpy.abs")(x, /) | Alias of [`jax.numpy.absolute()`](_autosummary/jax.numpy.absolute.html#jax.numpy.absolute "jax.numpy.absolute"). |
| [`absolute`](_autosummary/jax.numpy.absolute.html#jax.numpy.absolute "jax.numpy.absolute")(x, /) | Calculate the absolute value element-wise. |
| [`acos`](_autosummary/jax.numpy.acos.html#jax.numpy.acos "jax.numpy.acos")(x, /) | Alias of [`jax.numpy.arccos()`](_autosummary/jax.numpy.arccos.html#jax.numpy.arccos "jax.numpy.arccos") |
| [`acosh`](_autosummary/jax.numpy.acosh.html#jax.numpy.acosh "jax.numpy.acosh")(x, /) | Alias of [`jax.numpy.arccosh()`](_autosummary/jax.numpy.arccosh.html#jax.numpy.arccosh "jax.numpy.arccosh") |
| [`add`](_autosummary/jax.numpy.add.html#jax.numpy.add "jax.numpy.add") | Add two arrays element-wise. |
| [`all`](_autosummary/jax.numpy.all.html#jax.numpy.all "jax.numpy.all")(a\[, axis, out, keepdims, where\]) | Test whether all array elements along a given axis evaluate to True. |
| [`allclose`](_autosummary/jax.numpy.allclose.html#jax.numpy.allclose "jax.numpy.allclose")(a, b\[, rtol, atol, equal_nan\]) | Check if two arrays are element-wise approximately equal within a tolerance. |
| [`amax`](_autosummary/jax.numpy.amax.html#jax.numpy.amax "jax.numpy.amax")(a\[, axis, out, keepdims, initial, where\]) | Alias of [`jax.numpy.max()`](_autosummary/jax.numpy.max.html#jax.numpy.max "jax.numpy.max"). |
| [`amin`](_autosummary/jax.numpy.amin.html#jax.numpy.amin "jax.numpy.amin")(a\[, axis, out, keepdims, initial, where\]) | Alias of [`jax.numpy.min()`](_autosummary/jax.numpy.min.html#jax.numpy.min "jax.numpy.min"). |
| [`angle`](_autosummary/jax.numpy.angle.html#jax.numpy.angle "jax.numpy.angle")(z\[, deg\]) | Return the angle of a complex valued number or array. |
| [`any`](_autosummary/jax.numpy.any.html#jax.numpy.any "jax.numpy.any")(a\[, axis, out, keepdims, where\]) | Test whether any of the array elements along a given axis evaluate to True. |
| [`append`](_autosummary/jax.numpy.append.html#jax.numpy.append "jax.numpy.append")(arr, values\[, axis\]) | Return a new array with values appended to the end of the original array. |
| [`apply_along_axis`](_autosummary/jax.numpy.apply_along_axis.html#jax.numpy.apply_along_axis "jax.numpy.apply_along_axis")(func1d, axis, arr, \*args, ...) | Apply a function to 1D array slices along an axis. |
| [`apply_over_axes`](_autosummary/jax.numpy.apply_over_axes.html#jax.numpy.apply_over_axes "jax.numpy.apply_over_axes")(func, a, axes) | Apply a function repeatedly over specified axes. |
| [`arange`](_autosummary/jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange")(start\[, stop, step, dtype, device, ...\]) | Create an array of evenly-spaced values. |
| [`arccos`](_autosummary/jax.numpy.arccos.html#jax.numpy.arccos "jax.numpy.arccos")(x, /) | Compute element-wise inverse of trigonometric cosine of input. |
| [`arccosh`](_autosummary/jax.numpy.arccosh.html#jax.numpy.arccosh "jax.numpy.arccosh")(x, /) | Calculate element-wise inverse of hyperbolic cosine of input. |
| [`arcsin`](_autosummary/jax.numpy.arcsin.html#jax.numpy.arcsin "jax.numpy.arcsin")(x, /) | Compute element-wise inverse of trigonometric sine of input. |
| [`arcsinh`](_autosummary/jax.numpy.arcsinh.html#jax.numpy.arcsinh "jax.numpy.arcsinh")(x, /) | Calculate element-wise inverse of hyperbolic sine of input. |
| [`arctan`](_autosummary/jax.numpy.arctan.html#jax.numpy.arctan "jax.numpy.arctan")(x, /) | Compute element-wise inverse of trigonometric tangent of input. |
| [`arctan2`](_autosummary/jax.numpy.arctan2.html#jax.numpy.arctan2 "jax.numpy.arctan2")(x1, x2, /) | Compute the arctangent of x1/x2, choosing the correct quadrant. |
| [`arctanh`](_autosummary/jax.numpy.arctanh.html#jax.numpy.arctanh "jax.numpy.arctanh")(x, /) | Calculate element-wise inverse of hyperbolic tangent of input. |
| [`argmax`](_autosummary/jax.numpy.argmax.html#jax.numpy.argmax "jax.numpy.argmax")(a\[, axis, out, keepdims\]) | Return the index of the maximum value of an array. |
| [`argmin`](_autosummary/jax.numpy.argmin.html#jax.numpy.argmin "jax.numpy.argmin")(a\[, axis, out, keepdims\]) | Return the index of the minimum value of an array. |
| [`argpartition`](_autosummary/jax.numpy.argpartition.html#jax.numpy.argpartition "jax.numpy.argpartition")(a, kth\[, axis\]) | Returns indices that partially sort an array. |
| [`argsort`](_autosummary/jax.numpy.argsort.html#jax.numpy.argsort "jax.numpy.argsort")(a\[, axis, kind, order, stable, ...\]) | Return indices that sort an array. |
| [`argwhere`](_autosummary/jax.numpy.argwhere.html#jax.numpy.argwhere "jax.numpy.argwhere")(a, \*\[, size, fill_value\]) | Find the indices of nonzero array elements |
| [`around`](_autosummary/jax.numpy.around.html#jax.numpy.around "jax.numpy.around")(a\[, decimals, out\]) | Alias of [`jax.numpy.round()`](_autosummary/jax.numpy.round.html#jax.numpy.round "jax.numpy.round") |
| [`array`](_autosummary/jax.numpy.array.html#jax.numpy.array "jax.numpy.array")(object\[, dtype, copy, order, ndmin, ...\]) | Convert an object to a JAX array. |
| [`array_equal`](_autosummary/jax.numpy.array_equal.html#jax.numpy.array_equal "jax.numpy.array_equal")(a1, a2\[, equal_nan\]) | Check if two arrays are element-wise equal. |
| [`array_equiv`](_autosummary/jax.numpy.array_equiv.html#jax.numpy.array_equiv "jax.numpy.array_equiv")(a1, a2) | Check if two arrays are element-wise equal. |
| [`array_repr`](_autosummary/jax.numpy.array_repr.html#jax.numpy.array_repr "jax.numpy.array_repr")(arr\[, max_line_width, precision, ...\]) | Return the string representation of an array. |
| [`array_split`](_autosummary/jax.numpy.array_split.html#jax.numpy.array_split "jax.numpy.array_split")(ary, indices_or_sections\[, axis\]) | Split an array into sub-arrays. |
| [`array_str`](_autosummary/jax.numpy.array_str.html#jax.numpy.array_str "jax.numpy.array_str")(a\[, max_line_width, precision, ...\]) | Return a string representation of the data in an array. |
| [`asarray`](_autosummary/jax.numpy.asarray.html#jax.numpy.asarray "jax.numpy.asarray")(a\[, dtype, order, copy, device, ...\]) | Convert an object to a JAX array. |
| [`asin`](_autosummary/jax.numpy.asin.html#jax.numpy.asin "jax.numpy.asin")(x, /) | Alias of [`jax.numpy.arcsin()`](_autosummary/jax.numpy.arcsin.html#jax.numpy.arcsin "jax.numpy.arcsin") |
| [`asinh`](_autosummary/jax.numpy.asinh.html#jax.numpy.asinh "jax.numpy.asinh")(x, /) | Alias of [`jax.numpy.arcsinh()`](_autosummary/jax.numpy.arcsinh.html#jax.numpy.arcsinh "jax.numpy.arcsinh") |
| [`astype`](_autosummary/jax.numpy.astype.html#jax.numpy.astype "jax.numpy.astype")(x, dtype, /, \*\[, copy, device\]) | Convert an array to a specified dtype. |
| [`atan`](_autosummary/jax.numpy.atan.html#jax.numpy.atan "jax.numpy.atan")(x, /) | Alias of [`jax.numpy.arctan()`](_autosummary/jax.numpy.arctan.html#jax.numpy.arctan "jax.numpy.arctan") |
| [`atanh`](_autosummary/jax.numpy.atanh.html#jax.numpy.atanh "jax.numpy.atanh")(x, /) | Alias of [`jax.numpy.arctanh()`](_autosummary/jax.numpy.arctanh.html#jax.numpy.arctanh "jax.numpy.arctanh") |
| [`atan2`](_autosummary/jax.numpy.atan2.html#jax.numpy.atan2 "jax.numpy.atan2")(x1, x2, /) | Alias of [`jax.numpy.arctan2()`](_autosummary/jax.numpy.arctan2.html#jax.numpy.arctan2 "jax.numpy.arctan2") |
| [`atleast_1d`](_autosummary/jax.numpy.atleast_1d.html#jax.numpy.atleast_1d "jax.numpy.atleast_1d")(\*arys) | Convert inputs to arrays with at least 1 dimension. |
| [`atleast_2d`](_autosummary/jax.numpy.atleast_2d.html#jax.numpy.atleast_2d "jax.numpy.atleast_2d")(\*arys) | Convert inputs to arrays with at least 2 dimensions. |
| [`atleast_3d`](_autosummary/jax.numpy.atleast_3d.html#jax.numpy.atleast_3d "jax.numpy.atleast_3d")(\*arys) | Convert inputs to arrays with at least 3 dimensions. |
| [`average`](_autosummary/jax.numpy.average.html#jax.numpy.average "jax.numpy.average")(a\[, axis, weights, returned, keepdims\]) | Compute the weighed average. |
| [`bartlett`](_autosummary/jax.numpy.bartlett.html#jax.numpy.bartlett "jax.numpy.bartlett")(M) | Return a Bartlett window of size M. |
| [`bincount`](_autosummary/jax.numpy.bincount.html#jax.numpy.bincount "jax.numpy.bincount")(x\[, weights, minlength, length, ...\]) | Count the number of occurrences of each value in an integer array. |
| [`bitwise_and`](_autosummary/jax.numpy.bitwise_and.html#jax.numpy.bitwise_and "jax.numpy.bitwise_and") | Compute the bitwise AND operation elementwise. |
| [`bitwise_count`](_autosummary/jax.numpy.bitwise_count.html#jax.numpy.bitwise_count "jax.numpy.bitwise_count")(x, /) | Counts the number of 1 bits in the binary representation of the absolute value of each element of `x`. |
| [`bitwise_invert`](_autosummary/jax.numpy.bitwise_invert.html#jax.numpy.bitwise_invert "jax.numpy.bitwise_invert")(x, /) | Alias of [`jax.numpy.invert()`](_autosummary/jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert"). |
| [`bitwise_left_shift`](_autosummary/jax.numpy.bitwise_left_shift.html#jax.numpy.bitwise_left_shift "jax.numpy.bitwise_left_shift")(x, y, /) | Alias of [`jax.numpy.left_shift()`](_autosummary/jax.numpy.left_shift.html#jax.numpy.left_shift "jax.numpy.left_shift"). |
| [`bitwise_not`](_autosummary/jax.numpy.bitwise_not.html#jax.numpy.bitwise_not "jax.numpy.bitwise_not")(x, /) | Alias of [`jax.numpy.invert()`](_autosummary/jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert"). |
| [`bitwise_or`](_autosummary/jax.numpy.bitwise_or.html#jax.numpy.bitwise_or "jax.numpy.bitwise_or") | Compute the bitwise OR operation elementwise. |
| [`bitwise_right_shift`](_autosummary/jax.numpy.bitwise_right_shift.html#jax.numpy.bitwise_right_shift "jax.numpy.bitwise_right_shift")(x1, x2, /) | Alias of [`jax.numpy.right_shift()`](_autosummary/jax.numpy.right_shift.html#jax.numpy.right_shift "jax.numpy.right_shift"). |
| [`bitwise_xor`](_autosummary/jax.numpy.bitwise_xor.html#jax.numpy.bitwise_xor "jax.numpy.bitwise_xor") | Compute the bitwise XOR operation elementwise. |
| [`blackman`](_autosummary/jax.numpy.blackman.html#jax.numpy.blackman "jax.numpy.blackman")(M) | Return a Blackman window of size M. |
| [`block`](_autosummary/jax.numpy.block.html#jax.numpy.block "jax.numpy.block")(arrays) | Create an array from a list of blocks. |
| [`bool_`](_autosummary/jax.numpy.bool_.html#jax.numpy.bool_ "jax.numpy.bool_") | alias of `bool` |
| [`broadcast_arrays`](_autosummary/jax.numpy.broadcast_arrays.html#jax.numpy.broadcast_arrays "jax.numpy.broadcast_arrays")(\*args) | Broadcast arrays to a common shape. |
| [`broadcast_shapes`](_autosummary/jax.numpy.broadcast_shapes.html#jax.numpy.broadcast_shapes "jax.numpy.broadcast_shapes")(\*shapes) | Broadcast input shapes to a common output shape. |
| [`broadcast_to`](_autosummary/jax.numpy.broadcast_to.html#jax.numpy.broadcast_to "jax.numpy.broadcast_to")(array, shape, \*\[, out_sharding\]) | Broadcast an array to a specified shape. |
| [`c_`](_autosummary/jax.numpy.c_.html#jax.numpy.c_ "jax.numpy.c_") | Concatenate slices, scalars and array-like objects along the last axis. |
| [`can_cast`](_autosummary/jax.numpy.can_cast.html#jax.numpy.can_cast "jax.numpy.can_cast")(from\_, to\[, casting\]) | Returns True if cast between data types can occur according to the casting rule. |
| [`cbrt`](_autosummary/jax.numpy.cbrt.html#jax.numpy.cbrt "jax.numpy.cbrt")(x, /) | Calculates element-wise cube root of the input array. |
| [`cdouble`](_autosummary/jax.numpy.cdouble.html#jax.numpy.cdouble "jax.numpy.cdouble") | alias of [`complex128`](_autosummary/jax.numpy.complex128.html#jax.numpy.complex128 "jax.numpy.complex128") |
| [`ceil`](_autosummary/jax.numpy.ceil.html#jax.numpy.ceil "jax.numpy.ceil")(x, /) | Round input to the nearest integer upwards. |
| [`character`](_autosummary/jax.numpy.character.html#jax.numpy.character "jax.numpy.character")() | Abstract base class of all character string scalar types. |
| [`choose`](_autosummary/jax.numpy.choose.html#jax.numpy.choose "jax.numpy.choose")(a, choices\[, out, mode\]) | Construct an array by stacking slices of choice arrays. |
| [`clip`](_autosummary/jax.numpy.clip.html#jax.numpy.clip "jax.numpy.clip")(\[arr, min, max\]) | Clip array values to a specified range. |
| [`column_stack`](_autosummary/jax.numpy.column_stack.html#jax.numpy.column_stack "jax.numpy.column_stack")(tup) | Stack arrays column-wise. |
| [`complex_`](_autosummary/jax.numpy.complex_.html#jax.numpy.complex_ "jax.numpy.complex_") | alias of [`complex128`](_autosummary/jax.numpy.complex128.html#jax.numpy.complex128 "jax.numpy.complex128") |
| [`complex128`](_autosummary/jax.numpy.complex128.html#jax.numpy.complex128 "jax.numpy.complex128")(x) | A JAX scalar constructor of type complex128. |
| [`complex64`](_autosummary/jax.numpy.complex64.html#jax.numpy.complex64 "jax.numpy.complex64")(x) | A JAX scalar constructor of type complex64. |
| [`complexfloating`](_autosummary/jax.numpy.complexfloating.html#jax.numpy.complexfloating "jax.numpy.complexfloating")() | Abstract base class of all complex number scalar types that are made up of floating-point numbers. |
| [`ComplexWarning`](_autosummary/jax.numpy.ComplexWarning.html#jax.numpy.ComplexWarning "jax.numpy.ComplexWarning") | The warning raised when casting a complex dtype to a real dtype. |
| [`compress`](_autosummary/jax.numpy.compress.html#jax.numpy.compress "jax.numpy.compress")(condition, a\[, axis, size, ...\]) | Compress an array along a given axis using a boolean condition. |
| [`concat`](_autosummary/jax.numpy.concat.html#jax.numpy.concat "jax.numpy.concat")(arrays, /, \*\[, axis\]) | Join arrays along an existing axis. |
| [`concatenate`](_autosummary/jax.numpy.concatenate.html#jax.numpy.concatenate "jax.numpy.concatenate")(arrays\[, axis, dtype\]) | Join arrays along an existing axis. |
| [`conj`](_autosummary/jax.numpy.conj.html#jax.numpy.conj "jax.numpy.conj")(x, /) | Alias of [`jax.numpy.conjugate()`](_autosummary/jax.numpy.conjugate.html#jax.numpy.conjugate "jax.numpy.conjugate") |
| [`conjugate`](_autosummary/jax.numpy.conjugate.html#jax.numpy.conjugate "jax.numpy.conjugate")(x, /) | Return element-wise complex-conjugate of the input. |
| [`convolve`](_autosummary/jax.numpy.convolve.html#jax.numpy.convolve "jax.numpy.convolve")(a, v\[, mode, precision, ...\]) | Convolution of two one dimensional arrays. |
| [`copy`](_autosummary/jax.numpy.copy.html#jax.numpy.copy "jax.numpy.copy")(a\[, order\]) | Return a copy of the array. |
| [`copysign`](_autosummary/jax.numpy.copysign.html#jax.numpy.copysign "jax.numpy.copysign")(x1, x2, /) | Copies the sign of each element in `x2` to the corresponding element in `x1`. |
| [`corrcoef`](_autosummary/jax.numpy.corrcoef.html#jax.numpy.corrcoef "jax.numpy.corrcoef")(x\[, y, rowvar, dtype\]) | Compute the Pearson correlation coefficients. |
| [`correlate`](_autosummary/jax.numpy.correlate.html#jax.numpy.correlate "jax.numpy.correlate")(a, v\[, mode, precision, ...\]) | Correlation of two one dimensional arrays. |
| [`cos`](_autosummary/jax.numpy.cos.html#jax.numpy.cos "jax.numpy.cos")(x, /) | Compute a trigonometric cosine of each element of input. |
| [`cosh`](_autosummary/jax.numpy.cosh.html#jax.numpy.cosh "jax.numpy.cosh")(x, /) | Calculate element-wise hyperbolic cosine of input. |
| [`count_nonzero`](_autosummary/jax.numpy.count_nonzero.html#jax.numpy.count_nonzero "jax.numpy.count_nonzero")(a\[, axis, keepdims\]) | Return the number of nonzero elements along a given axis. |
| [`cov`](_autosummary/jax.numpy.cov.html#jax.numpy.cov "jax.numpy.cov")(m\[, y, rowvar, bias, ddof, fweights, ...\]) | Estimate the weighted sample covariance. |
| [`cross`](_autosummary/jax.numpy.cross.html#jax.numpy.cross "jax.numpy.cross")(a, b\[, axisa, axisb, axisc, axis\]) | Compute the (batched) cross product of two arrays. |
| [`csingle`](_autosummary/jax.numpy.csingle.html#jax.numpy.csingle "jax.numpy.csingle") | alias of [`complex64`](_autosummary/jax.numpy.complex64.html#jax.numpy.complex64 "jax.numpy.complex64") |
| [`cumprod`](_autosummary/jax.numpy.cumprod.html#jax.numpy.cumprod "jax.numpy.cumprod")(a\[, axis, dtype, out\]) | Cumulative product of elements along an axis. |
| [`cumsum`](_autosummary/jax.numpy.cumsum.html#jax.numpy.cumsum "jax.numpy.cumsum")(a\[, axis, dtype, out\]) | Cumulative sum of elements along an axis. |
| [`cumulative_prod`](_autosummary/jax.numpy.cumulative_prod.html#jax.numpy.cumulative_prod "jax.numpy.cumulative_prod")(x, /, \*\[, axis, dtype, ...\]) | Cumulative product along the axis of an array. |
| [`cumulative_sum`](_autosummary/jax.numpy.cumulative_sum.html#jax.numpy.cumulative_sum "jax.numpy.cumulative_sum")(x, /, \*\[, axis, dtype, ...\]) | Cumulative sum along the axis of an array. |
| [`deg2rad`](_autosummary/jax.numpy.deg2rad.html#jax.numpy.deg2rad "jax.numpy.deg2rad")(x, /) | Convert angles from degrees to radians. |
| [`degrees`](_autosummary/jax.numpy.degrees.html#jax.numpy.degrees "jax.numpy.degrees")(x, /) | Alias of [`jax.numpy.rad2deg()`](_autosummary/jax.numpy.rad2deg.html#jax.numpy.rad2deg "jax.numpy.rad2deg") |
| [`delete`](_autosummary/jax.numpy.delete.html#jax.numpy.delete "jax.numpy.delete")(arr, obj\[, axis, assume_unique_indices\]) | Delete entry or entries from an array. |
| [`diag`](_autosummary/jax.numpy.diag.html#jax.numpy.diag "jax.numpy.diag")(v\[, k\]) | Returns the specified diagonal or constructs a diagonal array. |
| [`diag_indices`](_autosummary/jax.numpy.diag_indices.html#jax.numpy.diag_indices "jax.numpy.diag_indices")(n\[, ndim\]) | Return indices for accessing the main diagonal of a multidimensional array. |
| [`diag_indices_from`](_autosummary/jax.numpy.diag_indices_from.html#jax.numpy.diag_indices_from "jax.numpy.diag_indices_from")(arr) | Return indices for accessing the main diagonal of a given array. |
| [`diagflat`](_autosummary/jax.numpy.diagflat.html#jax.numpy.diagflat "jax.numpy.diagflat")(v\[, k\]) | Return a 2-D array with the flattened input array laid out on the diagonal. |
| [`diagonal`](_autosummary/jax.numpy.diagonal.html#jax.numpy.diagonal "jax.numpy.diagonal")(a\[, offset, axis1, axis2\]) | Returns the specified diagonal of an array. |
| [`diff`](_autosummary/jax.numpy.diff.html#jax.numpy.diff "jax.numpy.diff")(a\[, n, axis, prepend, append\]) | Calculate n-th order difference between array elements along a given axis. |
| [`digitize`](_autosummary/jax.numpy.digitize.html#jax.numpy.digitize "jax.numpy.digitize")(x, bins\[, right, method\]) | Convert an array to bin indices. |
| [`divide`](_autosummary/jax.numpy.divide.html#jax.numpy.divide "jax.numpy.divide")(x1, x2, /) | Alias of [`jax.numpy.true_divide()`](_autosummary/jax.numpy.true_divide.html#jax.numpy.true_divide "jax.numpy.true_divide"). |
| [`divmod`](_autosummary/jax.numpy.divmod.html#jax.numpy.divmod "jax.numpy.divmod")(x1, x2, /) | Calculates the integer quotient and remainder of x1 by x2 element-wise |
| [`dot`](_autosummary/jax.numpy.dot.html#jax.numpy.dot "jax.numpy.dot")(a, b, \*\[, precision, ...\]) | Compute the dot product of two arrays. |
| [`double`](_autosummary/jax.numpy.double.html#jax.numpy.double "jax.numpy.double") | alias of [`float64`](_autosummary/jax.numpy.float64.html#jax.numpy.float64 "jax.numpy.float64") |
| [`dsplit`](_autosummary/jax.numpy.dsplit.html#jax.numpy.dsplit "jax.numpy.dsplit")(ary, indices_or_sections) | Split an array into sub-arrays depth-wise. |
| [`dstack`](_autosummary/jax.numpy.dstack.html#jax.numpy.dstack "jax.numpy.dstack")(tup\[, dtype\]) | Stack arrays depth-wise. |
| [`dtype`](_autosummary/jax.numpy.dtype.html#jax.numpy.dtype "jax.numpy.dtype")(dtype\[, align, copy\]) |  |
| [`ediff1d`](_autosummary/jax.numpy.ediff1d.html#jax.numpy.ediff1d "jax.numpy.ediff1d")(ary\[, to_end, to_begin\]) | Compute the differences of the elements of the flattened array. |
| [`einsum`](_autosummary/jax.numpy.einsum.html#jax.numpy.einsum "jax.numpy.einsum")(subscripts, /, \*operands\[, out, ...\]) | Einstein summation |
| [`einsum_path`](_autosummary/jax.numpy.einsum_path.html#jax.numpy.einsum_path "jax.numpy.einsum_path")(subscripts, /, \*operands\[, optimize\]) | Evaluates the optimal contraction path without evaluating the einsum. |
| [`empty`](_autosummary/jax.numpy.empty.html#jax.numpy.empty "jax.numpy.empty")(shape\[, dtype, device, out_sharding\]) | Create an empty array. |
| [`empty_like`](_autosummary/jax.numpy.empty_like.html#jax.numpy.empty_like "jax.numpy.empty_like")(prototype\[, dtype, shape, device\]) | Create an empty array with the same shape and dtype as an array. |
| [`equal`](_autosummary/jax.numpy.equal.html#jax.numpy.equal "jax.numpy.equal")(x, y, /) | Returns element-wise truth value of `x`` ``==`` ``y`. |
| [`exp`](_autosummary/jax.numpy.exp.html#jax.numpy.exp "jax.numpy.exp")(x, /) | Calculate element-wise exponential of the input. |
| [`exp2`](_autosummary/jax.numpy.exp2.html#jax.numpy.exp2 "jax.numpy.exp2")(x, /) | Calculate element-wise base-2 exponential of input. |
| [`expand_dims`](_autosummary/jax.numpy.expand_dims.html#jax.numpy.expand_dims "jax.numpy.expand_dims")(a, axis) | Insert dimensions of length 1 into array |
| [`expm1`](_autosummary/jax.numpy.expm1.html#jax.numpy.expm1 "jax.numpy.expm1")(x, /) | Calculate `exp(x)-1` of each element of the input. |
| [`extract`](_autosummary/jax.numpy.extract.html#jax.numpy.extract "jax.numpy.extract")(condition, arr, \*\[, size, fill_value\]) | Return the elements of an array that satisfy a condition. |
| [`eye`](_autosummary/jax.numpy.eye.html#jax.numpy.eye "jax.numpy.eye")(N\[, M, k, dtype, device\]) | Create a square or rectangular identity matrix |
| [`fabs`](_autosummary/jax.numpy.fabs.html#jax.numpy.fabs "jax.numpy.fabs")(x, /) | Compute the element-wise absolute values of the real-valued input. |
| [`fill_diagonal`](_autosummary/jax.numpy.fill_diagonal.html#jax.numpy.fill_diagonal "jax.numpy.fill_diagonal")(a, val\[, wrap, inplace\]) | Return a copy of the array with the diagonal overwritten. |
| [`finfo`](_autosummary/jax.numpy.finfo.html#jax.numpy.finfo "jax.numpy.finfo")(dtype) | Machine limits for floating point types. |
| [`flatnonzero`](_autosummary/jax.numpy.flatnonzero.html#jax.numpy.flatnonzero "jax.numpy.flatnonzero")(a, \*\[, size, fill_value\]) | Return indices of nonzero elements in a flattened array |
| [`flexible`](_autosummary/jax.numpy.flexible.html#jax.numpy.flexible "jax.numpy.flexible")() | Abstract base class of all scalar types without predefined length. |
| [`flip`](_autosummary/jax.numpy.flip.html#jax.numpy.flip "jax.numpy.flip")(m\[, axis\]) | Reverse the order of elements of an array along the given axis. |
| [`fliplr`](_autosummary/jax.numpy.fliplr.html#jax.numpy.fliplr "jax.numpy.fliplr")(m) | Reverse the order of elements of an array along axis 1. |
| [`flipud`](_autosummary/jax.numpy.flipud.html#jax.numpy.flipud "jax.numpy.flipud")(m) | Reverse the order of elements of an array along axis 0. |
| [`float_`](_autosummary/jax.numpy.float_.html#jax.numpy.float_ "jax.numpy.float_") | alias of [`float64`](_autosummary/jax.numpy.float64.html#jax.numpy.float64 "jax.numpy.float64") |
| [`float_power`](_autosummary/jax.numpy.float_power.html#jax.numpy.float_power "jax.numpy.float_power")(x, y, /) | Calculate element-wise base `x` exponential of `y`. |
| [`float16`](_autosummary/jax.numpy.float16.html#jax.numpy.float16 "jax.numpy.float16")(x) | A JAX scalar constructor of type float16. |
| [`float32`](_autosummary/jax.numpy.float32.html#jax.numpy.float32 "jax.numpy.float32")(x) | A JAX scalar constructor of type float32. |
| [`float64`](_autosummary/jax.numpy.float64.html#jax.numpy.float64 "jax.numpy.float64")(x) | A JAX scalar constructor of type float64. |
| [`floating`](_autosummary/jax.numpy.floating.html#jax.numpy.floating "jax.numpy.floating")() | Abstract base class of all floating-point scalar types. |
| [`floor`](_autosummary/jax.numpy.floor.html#jax.numpy.floor "jax.numpy.floor")(x, /) | Round input to the nearest integer downwards. |
| [`floor_divide`](_autosummary/jax.numpy.floor_divide.html#jax.numpy.floor_divide "jax.numpy.floor_divide")(x1, x2, /) | Calculates the floor division of x1 by x2 element-wise |
| [`fmax`](_autosummary/jax.numpy.fmax.html#jax.numpy.fmax "jax.numpy.fmax")(x1, x2) | Return element-wise maximum of the input arrays. |
| [`fmin`](_autosummary/jax.numpy.fmin.html#jax.numpy.fmin "jax.numpy.fmin")(x1, x2) | Return element-wise minimum of the input arrays. |
| [`fmod`](_autosummary/jax.numpy.fmod.html#jax.numpy.fmod "jax.numpy.fmod")(x1, x2, /) | Calculate element-wise floating-point modulo operation. |
| [`frexp`](_autosummary/jax.numpy.frexp.html#jax.numpy.frexp "jax.numpy.frexp")(x, /) | Split floating point values into mantissa and twos exponent. |
| [`frombuffer`](_autosummary/jax.numpy.frombuffer.html#jax.numpy.frombuffer "jax.numpy.frombuffer")(buffer\[, dtype, count, offset\]) | Convert a buffer into a 1-D JAX array. |
| [`fromfile`](_autosummary/jax.numpy.fromfile.html#jax.numpy.fromfile "jax.numpy.fromfile")(\*args, \*\*kwargs) | Unimplemented JAX wrapper for jnp.fromfile. |
| [`fromfunction`](_autosummary/jax.numpy.fromfunction.html#jax.numpy.fromfunction "jax.numpy.fromfunction")(function, shape, \*\[, dtype\]) | Create an array from a function applied over indices. |
| [`fromiter`](_autosummary/jax.numpy.fromiter.html#jax.numpy.fromiter "jax.numpy.fromiter")(\*args, \*\*kwargs) | Unimplemented JAX wrapper for jnp.fromiter. |
| [`frompyfunc`](_autosummary/jax.numpy.frompyfunc.html#jax.numpy.frompyfunc "jax.numpy.frompyfunc")(func, /, nin, nout, \*\[, identity\]) | Create a JAX ufunc from an arbitrary JAX-compatible scalar function. |
| [`fromstring`](_autosummary/jax.numpy.fromstring.html#jax.numpy.fromstring "jax.numpy.fromstring")(string\[, dtype, count\]) | Convert a string of text into 1-D JAX array. |
| [`from_dlpack`](_autosummary/jax.numpy.from_dlpack.html#jax.numpy.from_dlpack "jax.numpy.from_dlpack")(x, /, \*\[, device, copy\]) | Construct a JAX array via DLPack. |
| [`full`](_autosummary/jax.numpy.full.html#jax.numpy.full "jax.numpy.full")(shape, fill_value\[, dtype, device, ...\]) | Create an array full of a specified value. |
| [`full_like`](_autosummary/jax.numpy.full_like.html#jax.numpy.full_like "jax.numpy.full_like")(a, fill_value\[, dtype, shape, ...\]) | Create an array full of a specified value with the same shape and dtype as an array. |
| [`gcd`](_autosummary/jax.numpy.gcd.html#jax.numpy.gcd "jax.numpy.gcd")(x1, x2) | Compute the greatest common divisor of two arrays. |
| [`generic`](_autosummary/jax.numpy.generic.html#jax.numpy.generic "jax.numpy.generic")() | Base class for numpy scalar types. |
| [`geomspace`](_autosummary/jax.numpy.geomspace.html#jax.numpy.geomspace "jax.numpy.geomspace")(start, stop\[, num, endpoint, ...\]) | Generate geometrically-spaced values. |
| [`get_printoptions`](_autosummary/jax.numpy.get_printoptions.html#jax.numpy.get_printoptions "jax.numpy.get_printoptions")() | Alias of [`numpy.get_printoptions()`](https://numpy.org/doc/stable/reference/generated/numpy.get_printoptions.html#numpy.get_printoptions "(in NumPy v2.4)"). |
| [`gradient`](_autosummary/jax.numpy.gradient.html#jax.numpy.gradient "jax.numpy.gradient")(f, \*varargs\[, axis, edge_order\]) | Compute the numerical gradient of a sampled function. |
| [`greater`](_autosummary/jax.numpy.greater.html#jax.numpy.greater "jax.numpy.greater")(x, y, /) | Return element-wise truth value of `x`` ``>`` ``y`. |
| [`greater_equal`](_autosummary/jax.numpy.greater_equal.html#jax.numpy.greater_equal "jax.numpy.greater_equal")(x, y, /) | Return element-wise truth value of `x`` ``>=`` ``y`. |
| [`hamming`](_autosummary/jax.numpy.hamming.html#jax.numpy.hamming "jax.numpy.hamming")(M) | Return a Hamming window of size M. |
| [`hanning`](_autosummary/jax.numpy.hanning.html#jax.numpy.hanning "jax.numpy.hanning")(M) | Return a Hanning window of size M. |
| [`heaviside`](_autosummary/jax.numpy.heaviside.html#jax.numpy.heaviside "jax.numpy.heaviside")(x1, x2, /) | Compute the heaviside step function. |
| [`histogram`](_autosummary/jax.numpy.histogram.html#jax.numpy.histogram "jax.numpy.histogram")(a\[, bins, range, weights, density\]) | Compute a 1-dimensional histogram. |
| [`histogram_bin_edges`](_autosummary/jax.numpy.histogram_bin_edges.html#jax.numpy.histogram_bin_edges "jax.numpy.histogram_bin_edges")(a\[, bins, range, weights\]) | Compute the bin edges for a histogram. |
| [`histogram2d`](_autosummary/jax.numpy.histogram2d.html#jax.numpy.histogram2d "jax.numpy.histogram2d")(x, y\[, bins, range, weights, ...\]) | Compute a 2-dimensional histogram. |
| [`histogramdd`](_autosummary/jax.numpy.histogramdd.html#jax.numpy.histogramdd "jax.numpy.histogramdd")(sample\[, bins, range, weights, ...\]) | Compute an N-dimensional histogram. |
| [`hsplit`](_autosummary/jax.numpy.hsplit.html#jax.numpy.hsplit "jax.numpy.hsplit")(ary, indices_or_sections) | Split an array into sub-arrays horizontally. |
| [`hstack`](_autosummary/jax.numpy.hstack.html#jax.numpy.hstack "jax.numpy.hstack")(tup\[, dtype\]) | Horizontally stack arrays. |
| [`hypot`](_autosummary/jax.numpy.hypot.html#jax.numpy.hypot "jax.numpy.hypot")(x1, x2, /) | Return element-wise hypotenuse for the given legs of a right angle triangle. |
| [`i0`](_autosummary/jax.numpy.i0.html#jax.numpy.i0 "jax.numpy.i0")(x) | Calculate modified Bessel function of first kind, zeroth order. |
| [`identity`](_autosummary/jax.numpy.identity.html#jax.numpy.identity "jax.numpy.identity")(n\[, dtype\]) | Create a square identity matrix |
| [`iinfo`](_autosummary/jax.numpy.iinfo.html#jax.numpy.iinfo "jax.numpy.iinfo")(int_type) |  |
| [`imag`](_autosummary/jax.numpy.imag.html#jax.numpy.imag "jax.numpy.imag")(val, /) | Return element-wise imaginary of part of the complex argument. |
| [`index_exp`](_autosummary/jax.numpy.index_exp.html#jax.numpy.index_exp "jax.numpy.index_exp") | A nicer way to build up index tuples for arrays. |
| [`indices`](_autosummary/jax.numpy.indices.html#jax.numpy.indices "jax.numpy.indices")(dimensions\[, dtype, sparse\]) | Generate arrays of grid indices. |
| [`inexact`](_autosummary/jax.numpy.inexact.html#jax.numpy.inexact "jax.numpy.inexact")() | Abstract base class of all numeric scalar types with a (potentially) inexact representation of the values in its range, such as floating-point numbers. |
| [`inner`](_autosummary/jax.numpy.inner.html#jax.numpy.inner "jax.numpy.inner")(a, b, \*\[, precision, ...\]) | Compute the inner product of two arrays. |
| [`insert`](_autosummary/jax.numpy.insert.html#jax.numpy.insert "jax.numpy.insert")(arr, obj, values\[, axis\]) | Insert entries into an array at specified indices. |
| [`int_`](_autosummary/jax.numpy.int_.html#jax.numpy.int_ "jax.numpy.int_") | alias of [`int64`](_autosummary/jax.numpy.int64.html#jax.numpy.int64 "jax.numpy.int64") |
| [`int16`](_autosummary/jax.numpy.int16.html#jax.numpy.int16 "jax.numpy.int16")(x) | A JAX scalar constructor of type int16. |
| [`int32`](_autosummary/jax.numpy.int32.html#jax.numpy.int32 "jax.numpy.int32")(x) | A JAX scalar constructor of type int32. |
| [`int64`](_autosummary/jax.numpy.int64.html#jax.numpy.int64 "jax.numpy.int64")(x) | A JAX scalar constructor of type int64. |
| [`int8`](_autosummary/jax.numpy.int8.html#jax.numpy.int8 "jax.numpy.int8")(x) | A JAX scalar constructor of type int8. |
| [`integer`](_autosummary/jax.numpy.integer.html#jax.numpy.integer "jax.numpy.integer")() | Abstract base class of all integer scalar types. |
| [`interp`](_autosummary/jax.numpy.interp.html#jax.numpy.interp "jax.numpy.interp")(x, xp, fp\[, left, right, period\]) | One-dimensional linear interpolation. |
| [`intersect1d`](_autosummary/jax.numpy.intersect1d.html#jax.numpy.intersect1d "jax.numpy.intersect1d")(ar1, ar2\[, assume_unique, ...\]) | Compute the set intersection of two 1D arrays. |
| [`invert`](_autosummary/jax.numpy.invert.html#jax.numpy.invert "jax.numpy.invert")(x, /) | Compute the bitwise inversion of an input. |
| [`isclose`](_autosummary/jax.numpy.isclose.html#jax.numpy.isclose "jax.numpy.isclose")(a, b\[, rtol, atol, equal_nan\]) | Check if the elements of two arrays are approximately equal within a tolerance. |
| [`iscomplex`](_autosummary/jax.numpy.iscomplex.html#jax.numpy.iscomplex "jax.numpy.iscomplex")(x) | Return boolean array showing where the input is complex. |
| [`iscomplexobj`](_autosummary/jax.numpy.iscomplexobj.html#jax.numpy.iscomplexobj "jax.numpy.iscomplexobj")(x) | Check if the input is a complex number or an array containing complex elements. |
| [`isdtype`](_autosummary/jax.numpy.isdtype.html#jax.numpy.isdtype "jax.numpy.isdtype")(dtype, kind) | Returns a boolean indicating whether a provided dtype is of a specified kind. |
| [`isfinite`](_autosummary/jax.numpy.isfinite.html#jax.numpy.isfinite "jax.numpy.isfinite")(x, /) | Return a boolean array indicating whether each element of input is finite. |
| [`isin`](_autosummary/jax.numpy.isin.html#jax.numpy.isin "jax.numpy.isin")(element, test_elements\[, ...\]) | Determine whether elements in `element` appear in `test_elements`. |
| [`isinf`](_autosummary/jax.numpy.isinf.html#jax.numpy.isinf "jax.numpy.isinf")(x, /) | Return a boolean array indicating whether each element of input is infinite. |
| [`isnan`](_autosummary/jax.numpy.isnan.html#jax.numpy.isnan "jax.numpy.isnan")(x, /) | Returns a boolean array indicating whether each element of input is `NaN`. |
| [`isneginf`](_autosummary/jax.numpy.isneginf.html#jax.numpy.isneginf "jax.numpy.isneginf")(x, /\[, out\]) | Return boolean array indicating whether each element of input is negative infinite. |
| [`isposinf`](_autosummary/jax.numpy.isposinf.html#jax.numpy.isposinf "jax.numpy.isposinf")(x, /\[, out\]) | Return boolean array indicating whether each element of input is positive infinite. |
| [`isreal`](_autosummary/jax.numpy.isreal.html#jax.numpy.isreal "jax.numpy.isreal")(x) | Return boolean array showing where the input is real. |
| [`isrealobj`](_autosummary/jax.numpy.isrealobj.html#jax.numpy.isrealobj "jax.numpy.isrealobj")(x) | Check if the input is not a complex number or an array containing complex elements. |
| [`isscalar`](_autosummary/jax.numpy.isscalar.html#jax.numpy.isscalar "jax.numpy.isscalar")(element) | Return True if the input is a scalar. |
| [`issubdtype`](_autosummary/jax.numpy.issubdtype.html#jax.numpy.issubdtype "jax.numpy.issubdtype")(arg1, arg2) | Return True if arg1 is equal or lower than arg2 in the type hierarchy. |
| [`iterable`](_autosummary/jax.numpy.iterable.html#jax.numpy.iterable "jax.numpy.iterable")(y) | Check whether or not an object can be iterated over. |
| [`ix_`](_autosummary/jax.numpy.ix_.html#jax.numpy.ix_ "jax.numpy.ix_")(\*args) | Return a multi-dimensional grid (open mesh) from N one-dimensional sequences. |
| [`kaiser`](_autosummary/jax.numpy.kaiser.html#jax.numpy.kaiser "jax.numpy.kaiser")(M, beta) | Return a Kaiser window of size M. |
| [`kron`](_autosummary/jax.numpy.kron.html#jax.numpy.kron "jax.numpy.kron")(a, b) | Compute the Kronecker product of two input arrays. |
| [`lcm`](_autosummary/jax.numpy.lcm.html#jax.numpy.lcm "jax.numpy.lcm")(x1, x2) | Compute the least common multiple of two arrays. |
| [`ldexp`](_autosummary/jax.numpy.ldexp.html#jax.numpy.ldexp "jax.numpy.ldexp")(x1, x2, /) | Compute x1 \* 2 \*\* x2 |
| [`left_shift`](_autosummary/jax.numpy.left_shift.html#jax.numpy.left_shift "jax.numpy.left_shift")(x, y, /) | Shift bits of `x` to left by the amount specified in `y`, element-wise. |
| [`less`](_autosummary/jax.numpy.less.html#jax.numpy.less "jax.numpy.less")(x, y, /) | Return element-wise truth value of `x`` ``<`` ``y`. |
| [`less_equal`](_autosummary/jax.numpy.less_equal.html#jax.numpy.less_equal "jax.numpy.less_equal")(x, y, /) | Return element-wise truth value of `x`` ``<=`` ``y`. |
| [`lexsort`](_autosummary/jax.numpy.lexsort.html#jax.numpy.lexsort "jax.numpy.lexsort")(keys\[, axis\]) | Sort a sequence of keys in lexicographic order. |
| [`linspace`](_autosummary/jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace")(start, stop\[, num, endpoint, ...\]) | Return evenly-spaced numbers within an interval. |
| [`load`](_autosummary/jax.numpy.load.html#jax.numpy.load "jax.numpy.load")(file, \*args, \*\*kwargs) | Load JAX arrays from npy files. |
| [`log`](_autosummary/jax.numpy.log.html#jax.numpy.log "jax.numpy.log")(x, /) | Calculate element-wise natural logarithm of the input. |
| [`log10`](_autosummary/jax.numpy.log10.html#jax.numpy.log10 "jax.numpy.log10")(x, /) | Calculates the base-10 logarithm of x element-wise |
| [`log1p`](_autosummary/jax.numpy.log1p.html#jax.numpy.log1p "jax.numpy.log1p")(x, /) | Calculates element-wise logarithm of one plus input, `log(x+1)`. |
| [`log2`](_autosummary/jax.numpy.log2.html#jax.numpy.log2 "jax.numpy.log2")(x, /) | Calculates the base-2 logarithm of `x` element-wise. |
| [`logaddexp`](_autosummary/jax.numpy.logaddexp.html#jax.numpy.logaddexp "jax.numpy.logaddexp") | Compute `log(exp(x1)`` ``+`` ``exp(x2))` avoiding overflow. |
| [`logaddexp2`](_autosummary/jax.numpy.logaddexp2.html#jax.numpy.logaddexp2 "jax.numpy.logaddexp2") | Logarithm of the sum of exponentials of inputs in base-2 avoiding overflow. |
| [`logical_and`](_autosummary/jax.numpy.logical_and.html#jax.numpy.logical_and "jax.numpy.logical_and") | Compute the logical AND operation elementwise. |
| [`logical_not`](_autosummary/jax.numpy.logical_not.html#jax.numpy.logical_not "jax.numpy.logical_not")(x, /) | Compute NOT bool(x) element-wise. |
| [`logical_or`](_autosummary/jax.numpy.logical_or.html#jax.numpy.logical_or "jax.numpy.logical_or") | Compute the logical OR operation elementwise. |
| [`logical_xor`](_autosummary/jax.numpy.logical_xor.html#jax.numpy.logical_xor "jax.numpy.logical_xor") | Compute the logical XOR operation elementwise. |
| [`logspace`](_autosummary/jax.numpy.logspace.html#jax.numpy.logspace "jax.numpy.logspace")(start, stop\[, num, endpoint, base, ...\]) | Generate logarithmically-spaced values. |
| [`mask_indices`](_autosummary/jax.numpy.mask_indices.html#jax.numpy.mask_indices "jax.numpy.mask_indices")(n, mask_func\[, k, size\]) | Return indices of a mask of an (n, n) array. |
| [`matmul`](_autosummary/jax.numpy.matmul.html#jax.numpy.matmul "jax.numpy.matmul")(a, b, \*\[, precision, ...\]) | Perform a matrix multiplication. |
| [`matrix_transpose`](_autosummary/jax.numpy.matrix_transpose.html#jax.numpy.matrix_transpose "jax.numpy.matrix_transpose")(x, /) | Transpose the last two dimensions of an array. |
| [`matvec`](_autosummary/jax.numpy.matvec.html#jax.numpy.matvec "jax.numpy.matvec")(x1, x2, /) | Batched matrix-vector product. |
| [`max`](_autosummary/jax.numpy.max.html#jax.numpy.max "jax.numpy.max")(a\[, axis, out, keepdims, initial, where\]) | Return the maximum of the array elements along a given axis. |
| [`maximum`](_autosummary/jax.numpy.maximum.html#jax.numpy.maximum "jax.numpy.maximum") | Return element-wise maximum of the input arrays. |
| [`mean`](_autosummary/jax.numpy.mean.html#jax.numpy.mean "jax.numpy.mean")(a\[, axis, dtype, out, keepdims, where\]) | Return the mean of array elements along a given axis. |
| [`median`](_autosummary/jax.numpy.median.html#jax.numpy.median "jax.numpy.median")(a\[, axis, out, overwrite_input, keepdims\]) | Return the median of array elements along a given axis. |
| [`meshgrid`](_autosummary/jax.numpy.meshgrid.html#jax.numpy.meshgrid "jax.numpy.meshgrid")(\*xi\[, copy, sparse, indexing\]) | Construct N-dimensional grid arrays from N 1-dimensional vectors. |
| [`mgrid`](_autosummary/jax.numpy.mgrid.html#jax.numpy.mgrid "jax.numpy.mgrid") | Return dense multi-dimensional "meshgrid". |
| [`min`](_autosummary/jax.numpy.min.html#jax.numpy.min "jax.numpy.min")(a\[, axis, out, keepdims, initial, where\]) | Return the minimum of array elements along a given axis. |
| [`minimum`](_autosummary/jax.numpy.minimum.html#jax.numpy.minimum "jax.numpy.minimum") | Return element-wise minimum of the input arrays. |
| [`mod`](_autosummary/jax.numpy.mod.html#jax.numpy.mod "jax.numpy.mod")(x1, x2, /) | Alias of [`jax.numpy.remainder()`](_autosummary/jax.numpy.remainder.html#jax.numpy.remainder "jax.numpy.remainder") |
| [`modf`](_autosummary/jax.numpy.modf.html#jax.numpy.modf "jax.numpy.modf")(x, /\[, out\]) | Return element-wise fractional and integral parts of the input array. |
| [`moveaxis`](_autosummary/jax.numpy.moveaxis.html#jax.numpy.moveaxis "jax.numpy.moveaxis")(a, source, destination) | Move an array axis to a new position |
| [`multiply`](_autosummary/jax.numpy.multiply.html#jax.numpy.multiply "jax.numpy.multiply") | Multiply two arrays element-wise. |
| [`nan_to_num`](_autosummary/jax.numpy.nan_to_num.html#jax.numpy.nan_to_num "jax.numpy.nan_to_num")(x\[, copy, nan, posinf, neginf\]) | Replace NaN and infinite entries in an array. |
| [`nanargmax`](_autosummary/jax.numpy.nanargmax.html#jax.numpy.nanargmax "jax.numpy.nanargmax")(a\[, axis, out, keepdims\]) | Return the index of the maximum value of an array, ignoring NaNs. |
| [`nanargmin`](_autosummary/jax.numpy.nanargmin.html#jax.numpy.nanargmin "jax.numpy.nanargmin")(a\[, axis, out, keepdims\]) | Return the index of the minimum value of an array, ignoring NaNs. |
| [`nancumprod`](_autosummary/jax.numpy.nancumprod.html#jax.numpy.nancumprod "jax.numpy.nancumprod")(a\[, axis, dtype, out\]) | Cumulative product of elements along an axis, ignoring NaN values. |
| [`nancumsum`](_autosummary/jax.numpy.nancumsum.html#jax.numpy.nancumsum "jax.numpy.nancumsum")(a\[, axis, dtype, out\]) | Cumulative sum of elements along an axis, ignoring NaN values. |
| [`nanmax`](_autosummary/jax.numpy.nanmax.html#jax.numpy.nanmax "jax.numpy.nanmax")(a\[, axis, out, keepdims, initial, where\]) | Return the maximum of the array elements along a given axis, ignoring NaNs. |
| [`nanmean`](_autosummary/jax.numpy.nanmean.html#jax.numpy.nanmean "jax.numpy.nanmean")(a\[, axis, dtype, out, keepdims, where\]) | Return the mean of the array elements along a given axis, ignoring NaNs. |
| [`nanmedian`](_autosummary/jax.numpy.nanmedian.html#jax.numpy.nanmedian "jax.numpy.nanmedian")(a\[, axis, out, overwrite_input, ...\]) | Return the median of array elements along a given axis, ignoring NaNs. |
| [`nanmin`](_autosummary/jax.numpy.nanmin.html#jax.numpy.nanmin "jax.numpy.nanmin")(a\[, axis, out, keepdims, initial, where\]) | Return the minimum of the array elements along a given axis, ignoring NaNs. |
| [`nanpercentile`](_autosummary/jax.numpy.nanpercentile.html#jax.numpy.nanpercentile "jax.numpy.nanpercentile")(a, q\[, axis, out, ...\]) | Compute the percentile of the data along the specified axis, ignoring NaN values. |
| [`nanprod`](_autosummary/jax.numpy.nanprod.html#jax.numpy.nanprod "jax.numpy.nanprod")(a\[, axis, dtype, out, keepdims, ...\]) | Return the product of the array elements along a given axis, ignoring NaNs. |
| [`nanquantile`](_autosummary/jax.numpy.nanquantile.html#jax.numpy.nanquantile "jax.numpy.nanquantile")(a, q\[, axis, out, ...\]) | Compute the quantile of the data along the specified axis, ignoring NaNs. |
| [`nanstd`](_autosummary/jax.numpy.nanstd.html#jax.numpy.nanstd "jax.numpy.nanstd")(a\[, axis, dtype, out, ddof, ...\]) | Compute the standard deviation along a given axis, ignoring NaNs. |
| [`nansum`](_autosummary/jax.numpy.nansum.html#jax.numpy.nansum "jax.numpy.nansum")(a\[, axis, dtype, out, keepdims, ...\]) | Return the sum of the array elements along a given axis, ignoring NaNs. |
| [`nanvar`](_autosummary/jax.numpy.nanvar.html#jax.numpy.nanvar "jax.numpy.nanvar")(a\[, axis, dtype, out, ddof, ...\]) | Compute the variance of array elements along a given axis, ignoring NaNs. |
| [`ndarray`](_autosummary/jax.numpy.ndarray.html#jax.numpy.ndarray "jax.numpy.ndarray") | alias of [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") |
| [`ndim`](_autosummary/jax.numpy.ndim.html#jax.numpy.ndim "jax.numpy.ndim")(a) | Return the number of dimensions of an array. |
| [`negative`](_autosummary/jax.numpy.negative.html#jax.numpy.negative "jax.numpy.negative") | Return element-wise negative values of the input. |
| [`nextafter`](_autosummary/jax.numpy.nextafter.html#jax.numpy.nextafter "jax.numpy.nextafter")(x, y, /) | Return element-wise next floating point value after `x` towards `y`. |
| [`nonzero`](_autosummary/jax.numpy.nonzero.html#jax.numpy.nonzero "jax.numpy.nonzero")(a, \*\[, size, fill_value\]) | Return indices of nonzero elements of an array. |
| [`not_equal`](_autosummary/jax.numpy.not_equal.html#jax.numpy.not_equal "jax.numpy.not_equal")(x, y, /) | Returns element-wise truth value of `x`` ``!=`` ``y`. |
| [`number`](_autosummary/jax.numpy.number.html#jax.numpy.number "jax.numpy.number")() | Abstract base class of all numeric scalar types. |
| [`object_`](_autosummary/jax.numpy.object_.html#jax.numpy.object_ "jax.numpy.object_")(\[value\]) | Any Python object. |
| [`ogrid`](_autosummary/jax.numpy.ogrid.html#jax.numpy.ogrid "jax.numpy.ogrid") | Return open multi-dimensional "meshgrid". |
| [`ones`](_autosummary/jax.numpy.ones.html#jax.numpy.ones "jax.numpy.ones")(shape\[, dtype, device, out_sharding\]) | Create an array full of ones. |
| [`ones_like`](_autosummary/jax.numpy.ones_like.html#jax.numpy.ones_like "jax.numpy.ones_like")(a\[, dtype, shape, device, ...\]) | Create an array of ones with the same shape and dtype as an array. |
| [`outer`](_autosummary/jax.numpy.outer.html#jax.numpy.outer "jax.numpy.outer")(a, b\[, out\]) | Compute the outer product of two arrays. |
| [`packbits`](_autosummary/jax.numpy.packbits.html#jax.numpy.packbits "jax.numpy.packbits")(a\[, axis, bitorder\]) | Pack array of bits into a uint8 array. |
| [`pad`](_autosummary/jax.numpy.pad.html#jax.numpy.pad "jax.numpy.pad")(array, pad_width\[, mode\]) | Add padding to an array. |
| [`partition`](_autosummary/jax.numpy.partition.html#jax.numpy.partition "jax.numpy.partition")(a, kth\[, axis\]) | Returns a partially-sorted copy of an array. |
| [`percentile`](_autosummary/jax.numpy.percentile.html#jax.numpy.percentile "jax.numpy.percentile")(a, q\[, axis, out, ...\]) | Compute the percentile of the data along the specified axis. |
| [`permute_dims`](_autosummary/jax.numpy.permute_dims.html#jax.numpy.permute_dims "jax.numpy.permute_dims")(a, /, axes) | Permute the axes/dimensions of an array. |
| [`piecewise`](_autosummary/jax.numpy.piecewise.html#jax.numpy.piecewise "jax.numpy.piecewise")(x, condlist, funclist, \*args, \*\*kw) | Evaluate a function defined piecewise across the domain. |
| [`place`](_autosummary/jax.numpy.place.html#jax.numpy.place "jax.numpy.place")(arr, mask, vals, \*\[, inplace\]) | Update array elements based on a mask. |
| [`poly`](_autosummary/jax.numpy.poly.html#jax.numpy.poly "jax.numpy.poly")(seq_of_zeros) | Returns the coefficients of a polynomial for the given sequence of roots. |
| [`polyadd`](_autosummary/jax.numpy.polyadd.html#jax.numpy.polyadd "jax.numpy.polyadd")(a1, a2) | Returns the sum of the two polynomials. |
| [`polyder`](_autosummary/jax.numpy.polyder.html#jax.numpy.polyder "jax.numpy.polyder")(p\[, m\]) | Returns the coefficients of the derivative of specified order of a polynomial. |
| [`polydiv`](_autosummary/jax.numpy.polydiv.html#jax.numpy.polydiv "jax.numpy.polydiv")(u, v, \*\[, trim_leading_zeros\]) | Returns the quotient and remainder of polynomial division. |
| [`polyfit`](_autosummary/jax.numpy.polyfit.html#jax.numpy.polyfit "jax.numpy.polyfit")(x, y, deg\[, rcond, full, w, cov\]) | Least squares polynomial fit to data. |
| [`polyint`](_autosummary/jax.numpy.polyint.html#jax.numpy.polyint "jax.numpy.polyint")(p\[, m, k\]) | Returns the coefficients of the integration of specified order of a polynomial. |
| [`polymul`](_autosummary/jax.numpy.polymul.html#jax.numpy.polymul "jax.numpy.polymul")(a1, a2, \*\[, trim_leading_zeros\]) | Returns the product of two polynomials. |
| [`polysub`](_autosummary/jax.numpy.polysub.html#jax.numpy.polysub "jax.numpy.polysub")(a1, a2) | Returns the difference of two polynomials. |
| [`polyval`](_autosummary/jax.numpy.polyval.html#jax.numpy.polyval "jax.numpy.polyval")(p, x, \*\[, unroll\]) | Evaluates the polynomial at specific values. |
| [`positive`](_autosummary/jax.numpy.positive.html#jax.numpy.positive "jax.numpy.positive")(x, /) | Return element-wise positive values of the input. |
| [`pow`](_autosummary/jax.numpy.pow.html#jax.numpy.pow "jax.numpy.pow")(x1, x2, /) | Alias of [`jax.numpy.power()`](_autosummary/jax.numpy.power.html#jax.numpy.power "jax.numpy.power") |
| [`power`](_autosummary/jax.numpy.power.html#jax.numpy.power "jax.numpy.power")(x1, x2, /) | Calculate element-wise base `x1` exponential of `x2`. |
| [`printoptions`](_autosummary/jax.numpy.printoptions.html#jax.numpy.printoptions "jax.numpy.printoptions")(\*args, \*\*kwargs) | Alias of [`numpy.printoptions()`](https://numpy.org/doc/stable/reference/generated/numpy.printoptions.html#numpy.printoptions "(in NumPy v2.4)"). |
| [`prod`](_autosummary/jax.numpy.prod.html#jax.numpy.prod "jax.numpy.prod")(a\[, axis, dtype, out, keepdims, ...\]) | Return product of the array elements over a given axis. |
| [`promote_types`](_autosummary/jax.numpy.promote_types.html#jax.numpy.promote_types "jax.numpy.promote_types")(a, b) | Returns the type to which a binary operation should cast its arguments. |
| [`ptp`](_autosummary/jax.numpy.ptp.html#jax.numpy.ptp "jax.numpy.ptp")(a\[, axis, out, keepdims\]) | Return the peak-to-peak range along a given axis. |
| [`put`](_autosummary/jax.numpy.put.html#jax.numpy.put "jax.numpy.put")(a, ind, v\[, mode, inplace\]) | Put elements into an array at given indices. |
| [`put_along_axis`](_autosummary/jax.numpy.put_along_axis.html#jax.numpy.put_along_axis "jax.numpy.put_along_axis")(arr, indices, values, axis\[, ...\]) | Put values into the destination array by matching 1d index and data slices. |
| [`quantile`](_autosummary/jax.numpy.quantile.html#jax.numpy.quantile "jax.numpy.quantile")(a, q\[, axis, out, overwrite_input, ...\]) | Compute the quantile of the data along the specified axis. |
| [`r_`](_autosummary/jax.numpy.r_.html#jax.numpy.r_ "jax.numpy.r_") | Concatenate slices, scalars and array-like objects along the first axis. |
| [`rad2deg`](_autosummary/jax.numpy.rad2deg.html#jax.numpy.rad2deg "jax.numpy.rad2deg")(x, /) | Convert angles from radians to degrees. |
| [`radians`](_autosummary/jax.numpy.radians.html#jax.numpy.radians "jax.numpy.radians")(x, /) | Alias of [`jax.numpy.deg2rad()`](_autosummary/jax.numpy.deg2rad.html#jax.numpy.deg2rad "jax.numpy.deg2rad") |
| [`ravel`](_autosummary/jax.numpy.ravel.html#jax.numpy.ravel "jax.numpy.ravel")(a\[, order, out_sharding\]) | Flatten array into a 1-dimensional shape. |
| [`ravel_multi_index`](_autosummary/jax.numpy.ravel_multi_index.html#jax.numpy.ravel_multi_index "jax.numpy.ravel_multi_index")(multi_index, dims\[, mode, ...\]) | Convert multi-dimensional indices into flat indices. |
| [`real`](_autosummary/jax.numpy.real.html#jax.numpy.real "jax.numpy.real")(val, /) | Return element-wise real part of the complex argument. |
| [`reciprocal`](_autosummary/jax.numpy.reciprocal.html#jax.numpy.reciprocal "jax.numpy.reciprocal")(x, /) | Calculate element-wise reciprocal of the input. |
| [`remainder`](_autosummary/jax.numpy.remainder.html#jax.numpy.remainder "jax.numpy.remainder")(x1, x2, /) | Returns element-wise remainder of the division. |
| [`repeat`](_autosummary/jax.numpy.repeat.html#jax.numpy.repeat "jax.numpy.repeat")(a, repeats\[, axis, ...\]) | Construct an array from repeated elements. |
| [`reshape`](_autosummary/jax.numpy.reshape.html#jax.numpy.reshape "jax.numpy.reshape")(a, shape\[, order, copy, out_sharding\]) | Return a reshaped copy of an array. |
| [`resize`](_autosummary/jax.numpy.resize.html#jax.numpy.resize "jax.numpy.resize")(a, new_shape) | Return a new array with specified shape. |
| [`result_type`](_autosummary/jax.numpy.result_type.html#jax.numpy.result_type "jax.numpy.result_type")(\*args) | Return the result of applying JAX promotion rules to the inputs. |
| [`right_shift`](_autosummary/jax.numpy.right_shift.html#jax.numpy.right_shift "jax.numpy.right_shift")(x1, x2, /) | Right shift the bits of `x1` to the amount specified in `x2`. |
| [`rint`](_autosummary/jax.numpy.rint.html#jax.numpy.rint "jax.numpy.rint")(x, /) | Rounds the elements of x to the nearest integer |
| [`roll`](_autosummary/jax.numpy.roll.html#jax.numpy.roll "jax.numpy.roll")(a, shift\[, axis\]) | Roll the elements of an array along a specified axis. |
| [`rollaxis`](_autosummary/jax.numpy.rollaxis.html#jax.numpy.rollaxis "jax.numpy.rollaxis")(a, axis\[, start\]) | Roll the specified axis to a given position. |
| [`roots`](_autosummary/jax.numpy.roots.html#jax.numpy.roots "jax.numpy.roots")(p, \*\[, strip_zeros\]) | Returns the roots of a polynomial given the coefficients `p`. |
| [`rot90`](_autosummary/jax.numpy.rot90.html#jax.numpy.rot90 "jax.numpy.rot90")(m\[, k, axes\]) | Rotate an array by 90 degrees counterclockwise in the plane specified by axes. |
| [`round`](_autosummary/jax.numpy.round.html#jax.numpy.round "jax.numpy.round")(a\[, decimals, out\]) | Round input evenly to the given number of decimals. |
| [`s_`](_autosummary/jax.numpy.s_.html#jax.numpy.s_ "jax.numpy.s_") | A nicer way to build up index tuples for arrays. |
| [`save`](_autosummary/jax.numpy.save.html#jax.numpy.save "jax.numpy.save")(file, arr\[, allow_pickle\]) | Save an array to a binary file in NumPy `.npy` format. |
| [`savez`](_autosummary/jax.numpy.savez.html#jax.numpy.savez "jax.numpy.savez")(file, \*args\[, allow_pickle\]) | Save several arrays into a single file in uncompressed `.npz` format. |
| [`searchsorted`](_autosummary/jax.numpy.searchsorted.html#jax.numpy.searchsorted "jax.numpy.searchsorted")(a, v\[, side, sorter, method\]) | Perform a binary search within a sorted array. |
| [`select`](_autosummary/jax.numpy.select.html#jax.numpy.select "jax.numpy.select")(condlist, choicelist\[, default\]) | Select values based on a series of conditions. |
| [`set_printoptions`](_autosummary/jax.numpy.set_printoptions.html#jax.numpy.set_printoptions "jax.numpy.set_printoptions")(\*args, \*\*kwargs) | Alias of [`numpy.set_printoptions()`](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions "(in NumPy v2.4)"). |
| [`setdiff1d`](_autosummary/jax.numpy.setdiff1d.html#jax.numpy.setdiff1d "jax.numpy.setdiff1d")(ar1, ar2\[, assume_unique, size, ...\]) | Compute the set difference of two 1D arrays. |
| [`setxor1d`](_autosummary/jax.numpy.setxor1d.html#jax.numpy.setxor1d "jax.numpy.setxor1d")(ar1, ar2\[, assume_unique, size, ...\]) | Compute the set-wise xor of elements in two arrays. |
| [`shape`](_autosummary/jax.numpy.shape.html#jax.numpy.shape "jax.numpy.shape")(a) | Return the shape an array. |
| [`sign`](_autosummary/jax.numpy.sign.html#jax.numpy.sign "jax.numpy.sign")(x, /) | Return an element-wise indication of sign of the input. |
| [`signbit`](_autosummary/jax.numpy.signbit.html#jax.numpy.signbit "jax.numpy.signbit")(x, /) | Return the sign bit of array elements. |
| [`signedinteger`](_autosummary/jax.numpy.signedinteger.html#jax.numpy.signedinteger "jax.numpy.signedinteger")() | Abstract base class of all signed integer scalar types. |
| [`sin`](_autosummary/jax.numpy.sin.html#jax.numpy.sin "jax.numpy.sin")(x, /) | Compute a trigonometric sine of each element of input. |
| [`sinc`](_autosummary/jax.numpy.sinc.html#jax.numpy.sinc "jax.numpy.sinc")(x, /) | Calculate the normalized sinc function. |
| [`single`](_autosummary/jax.numpy.single.html#jax.numpy.single "jax.numpy.single") | alias of [`float32`](_autosummary/jax.numpy.float32.html#jax.numpy.float32 "jax.numpy.float32") |
| [`sinh`](_autosummary/jax.numpy.sinh.html#jax.numpy.sinh "jax.numpy.sinh")(x, /) | Calculate element-wise hyperbolic sine of input. |
| [`size`](_autosummary/jax.numpy.size.html#jax.numpy.size "jax.numpy.size")(a\[, axis\]) | Return number of elements along a given axis. |
| [`sort`](_autosummary/jax.numpy.sort.html#jax.numpy.sort "jax.numpy.sort")(a\[, axis, kind, order, stable, descending\]) | Return a sorted copy of an array. |
| [`sort_complex`](_autosummary/jax.numpy.sort_complex.html#jax.numpy.sort_complex "jax.numpy.sort_complex")(a) | Return a sorted copy of complex array. |
| [`spacing`](_autosummary/jax.numpy.spacing.html#jax.numpy.spacing "jax.numpy.spacing")(x, /) | Return the spacing between `x` and the next adjacent number. |
| [`split`](_autosummary/jax.numpy.split.html#jax.numpy.split "jax.numpy.split")(ary, indices_or_sections\[, axis\]) | Split an array into sub-arrays. |
| [`sqrt`](_autosummary/jax.numpy.sqrt.html#jax.numpy.sqrt "jax.numpy.sqrt")(x, /) | Calculates element-wise non-negative square root of the input array. |
| [`square`](_autosummary/jax.numpy.square.html#jax.numpy.square "jax.numpy.square")(x, /) | Calculate element-wise square of the input array. |
| [`squeeze`](_autosummary/jax.numpy.squeeze.html#jax.numpy.squeeze "jax.numpy.squeeze")(a\[, axis\]) | Remove one or more length-1 axes from array |
| [`stack`](_autosummary/jax.numpy.stack.html#jax.numpy.stack "jax.numpy.stack")(arrays\[, axis, out, dtype\]) | Join arrays along a new axis. |
| [`std`](_autosummary/jax.numpy.std.html#jax.numpy.std "jax.numpy.std")(a\[, axis, dtype, out, ddof, keepdims, ...\]) | Compute the standard deviation along a given axis. |
| [`subtract`](_autosummary/jax.numpy.subtract.html#jax.numpy.subtract "jax.numpy.subtract") | Subtract two arrays element-wise. |
| [`sum`](_autosummary/jax.numpy.sum.html#jax.numpy.sum "jax.numpy.sum")(a\[, axis, dtype, out, keepdims, ...\]) | Sum of the elements of the array over a given axis. |
| [`swapaxes`](_autosummary/jax.numpy.swapaxes.html#jax.numpy.swapaxes "jax.numpy.swapaxes")(a, axis1, axis2) | Swap two axes of an array. |
| [`take`](_autosummary/jax.numpy.take.html#jax.numpy.take "jax.numpy.take")(a, indices\[, axis, out, mode, ...\]) | Take elements from an array. |
| [`take_along_axis`](_autosummary/jax.numpy.take_along_axis.html#jax.numpy.take_along_axis "jax.numpy.take_along_axis")(arr, indices\[, axis, mode, ...\]) | Take elements from an array. |
| [`tan`](_autosummary/jax.numpy.tan.html#jax.numpy.tan "jax.numpy.tan")(x, /) | Compute a trigonometric tangent of each element of input. |
| [`tanh`](_autosummary/jax.numpy.tanh.html#jax.numpy.tanh "jax.numpy.tanh")(x, /) | Calculate element-wise hyperbolic tangent of input. |
| [`tensordot`](_autosummary/jax.numpy.tensordot.html#jax.numpy.tensordot "jax.numpy.tensordot")(a, b\[, axes, precision, ...\]) | Compute the tensor dot product of two N-dimensional arrays. |
| [`tile`](_autosummary/jax.numpy.tile.html#jax.numpy.tile "jax.numpy.tile")(A, reps) | Construct an array by repeating `A` along specified dimensions. |
| [`trace`](_autosummary/jax.numpy.trace.html#jax.numpy.trace "jax.numpy.trace")(a\[, offset, axis1, axis2, dtype, out\]) | Calculate sum of the diagonal of input along the given axes. |
| [`trapezoid`](_autosummary/jax.numpy.trapezoid.html#jax.numpy.trapezoid "jax.numpy.trapezoid")(y\[, x, dx, axis\]) | Integrate along the given axis using the composite trapezoidal rule. |
| [`transpose`](_autosummary/jax.numpy.transpose.html#jax.numpy.transpose "jax.numpy.transpose")(a\[, axes\]) | Return a transposed version of an N-dimensional array. |
| [`tri`](_autosummary/jax.numpy.tri.html#jax.numpy.tri "jax.numpy.tri")(N\[, M, k, dtype\]) | Return an array with ones on and below the diagonal and zeros elsewhere. |
| [`tril`](_autosummary/jax.numpy.tril.html#jax.numpy.tril "jax.numpy.tril")(m\[, k\]) | Return lower triangle of an array. |
| [`tril_indices`](_autosummary/jax.numpy.tril_indices.html#jax.numpy.tril_indices "jax.numpy.tril_indices")(n\[, k, m\]) | Return the indices of lower triangle of an array of size `(n,`` ``m)`. |
| [`tril_indices_from`](_autosummary/jax.numpy.tril_indices_from.html#jax.numpy.tril_indices_from "jax.numpy.tril_indices_from")(arr\[, k\]) | Return the indices of lower triangle of a given array. |
| [`trim_zeros`](_autosummary/jax.numpy.trim_zeros.html#jax.numpy.trim_zeros "jax.numpy.trim_zeros")(filt\[, trim, axis\]) | Trim leading and/or trailing zeros of the input array. |
| [`triu`](_autosummary/jax.numpy.triu.html#jax.numpy.triu "jax.numpy.triu")(m\[, k\]) | Return upper triangle of an array. |
| [`triu_indices`](_autosummary/jax.numpy.triu_indices.html#jax.numpy.triu_indices "jax.numpy.triu_indices")(n\[, k, m\]) | Return the indices of upper triangle of an array of size `(n,`` ``m)`. |
| [`triu_indices_from`](_autosummary/jax.numpy.triu_indices_from.html#jax.numpy.triu_indices_from "jax.numpy.triu_indices_from")(arr\[, k\]) | Return the indices of upper triangle of a given array. |
| [`true_divide`](_autosummary/jax.numpy.true_divide.html#jax.numpy.true_divide "jax.numpy.true_divide")(x1, x2, /) | Calculates the division of x1 by x2 element-wise |
| [`trunc`](_autosummary/jax.numpy.trunc.html#jax.numpy.trunc "jax.numpy.trunc")(x) | Round input to the nearest integer towards zero. |
| [`ufunc`](_autosummary/jax.numpy.ufunc.html#jax.numpy.ufunc "jax.numpy.ufunc")(func, /, nin, nout, \*\[, name, nargs, ...\]) | Universal functions which operation element-by-element on arrays. |
| [`uint`](_autosummary/jax.numpy.uint.html#jax.numpy.uint "jax.numpy.uint") | alias of [`uint64`](_autosummary/jax.numpy.uint64.html#jax.numpy.uint64 "jax.numpy.uint64") |
| [`uint16`](_autosummary/jax.numpy.uint16.html#jax.numpy.uint16 "jax.numpy.uint16")(x) | A JAX scalar constructor of type uint16. |
| [`uint32`](_autosummary/jax.numpy.uint32.html#jax.numpy.uint32 "jax.numpy.uint32")(x) | A JAX scalar constructor of type uint32. |
| [`uint64`](_autosummary/jax.numpy.uint64.html#jax.numpy.uint64 "jax.numpy.uint64")(x) | A JAX scalar constructor of type uint64. |
| [`uint8`](_autosummary/jax.numpy.uint8.html#jax.numpy.uint8 "jax.numpy.uint8")(x) | A JAX scalar constructor of type uint8. |
| [`union1d`](_autosummary/jax.numpy.union1d.html#jax.numpy.union1d "jax.numpy.union1d")(ar1, ar2, \*\[, size, fill_value\]) | Compute the set union of two 1D arrays. |
| [`unique`](_autosummary/jax.numpy.unique.html#jax.numpy.unique "jax.numpy.unique")(ar\[, return_index, return_inverse, ...\]) | Return the unique values from an array. |
| [`unique_all`](_autosummary/jax.numpy.unique_all.html#jax.numpy.unique_all "jax.numpy.unique_all")(x, /, \*\[, size, fill_value\]) | Return unique values from x, along with indices, inverse indices, and counts. |
| [`unique_counts`](_autosummary/jax.numpy.unique_counts.html#jax.numpy.unique_counts "jax.numpy.unique_counts")(x, /, \*\[, size, fill_value\]) | Return unique values from x, along with counts. |
| [`unique_inverse`](_autosummary/jax.numpy.unique_inverse.html#jax.numpy.unique_inverse "jax.numpy.unique_inverse")(x, /, \*\[, size, fill_value\]) | Return unique values from x, along with indices, inverse indices, and counts. |
| [`unique_values`](_autosummary/jax.numpy.unique_values.html#jax.numpy.unique_values "jax.numpy.unique_values")(x, /, \*\[, size, fill_value\]) | Return unique values from x, along with indices, inverse indices, and counts. |
| [`unpackbits`](_autosummary/jax.numpy.unpackbits.html#jax.numpy.unpackbits "jax.numpy.unpackbits")(a\[, axis, count, bitorder\]) | Unpack the bits in a uint8 array. |
| [`unravel_index`](_autosummary/jax.numpy.unravel_index.html#jax.numpy.unravel_index "jax.numpy.unravel_index")(indices, shape) | Convert flat indices into multi-dimensional indices. |
| [`unstack`](_autosummary/jax.numpy.unstack.html#jax.numpy.unstack "jax.numpy.unstack")(x, /, \*\[, axis\]) | Unstack an array along an axis. |
| [`unsignedinteger`](_autosummary/jax.numpy.unsignedinteger.html#jax.numpy.unsignedinteger "jax.numpy.unsignedinteger")() | Abstract base class of all unsigned integer scalar types. |
| [`unwrap`](_autosummary/jax.numpy.unwrap.html#jax.numpy.unwrap "jax.numpy.unwrap")(p\[, discont, axis, period\]) | Unwrap a periodic signal. |
| [`vander`](_autosummary/jax.numpy.vander.html#jax.numpy.vander "jax.numpy.vander")(x\[, N, increasing\]) | Generate a Vandermonde matrix. |
| [`var`](_autosummary/jax.numpy.var.html#jax.numpy.var "jax.numpy.var")(a\[, axis, dtype, out, ddof, keepdims, ...\]) | Compute the variance along a given axis. |
| [`vdot`](_autosummary/jax.numpy.vdot.html#jax.numpy.vdot "jax.numpy.vdot")(a, b, \*\[, precision, ...\]) | Perform a conjugate multiplication of two 1D vectors. |
| [`vecdot`](_autosummary/jax.numpy.vecdot.html#jax.numpy.vecdot "jax.numpy.vecdot")(x1, x2, /, \*\[, axis, precision, ...\]) | Perform a conjugate multiplication of two batched vectors. |
| [`vecmat`](_autosummary/jax.numpy.vecmat.html#jax.numpy.vecmat "jax.numpy.vecmat")(x1, x2, /) | Batched conjugate vector-matrix product. |
| [`vectorize`](_autosummary/jax.numpy.vectorize.html#jax.numpy.vectorize "jax.numpy.vectorize")(pyfunc, \*\[, excluded, signature\]) | Define a vectorized function with broadcasting. |
| [`vsplit`](_autosummary/jax.numpy.vsplit.html#jax.numpy.vsplit "jax.numpy.vsplit")(ary, indices_or_sections) | Split an array into sub-arrays vertically. |
| [`vstack`](_autosummary/jax.numpy.vstack.html#jax.numpy.vstack "jax.numpy.vstack")(tup\[, dtype\]) | Vertically stack arrays. |
| [`where`](_autosummary/jax.numpy.where.html#jax.numpy.where "jax.numpy.where")(condition\[, x, y, size, fill_value\]) | Select elements from two arrays based on a condition. |
| [`zeros`](_autosummary/jax.numpy.zeros.html#jax.numpy.zeros "jax.numpy.zeros")(shape\[, dtype, device, out_sharding\]) | Create an array full of zeros. |
| [`zeros_like`](_autosummary/jax.numpy.zeros_like.html#jax.numpy.zeros_like "jax.numpy.zeros_like")(a\[, dtype, shape, device, ...\]) | Create an array full of zeros with the same shape and dtype as an array. |

## jax.numpy.fft[\#](#module-jax.numpy.fft "Link to this heading")

|  |  |
|----|----|
| [`fft`](_autosummary/jax.numpy.fft.fft.html#jax.numpy.fft.fft "jax.numpy.fft.fft")(a\[, n, axis, norm\]) | Compute a one-dimensional discrete Fourier transform along a given axis. |
| [`fft2`](_autosummary/jax.numpy.fft.fft2.html#jax.numpy.fft.fft2 "jax.numpy.fft.fft2")(a\[, s, axes, norm\]) | Compute a two-dimensional discrete Fourier transform along given axes. |
| [`fftfreq`](_autosummary/jax.numpy.fft.fftfreq.html#jax.numpy.fft.fftfreq "jax.numpy.fft.fftfreq")(n\[, d, dtype, device\]) | Return sample frequencies for the discrete Fourier transform. |
| [`fftn`](_autosummary/jax.numpy.fft.fftn.html#jax.numpy.fft.fftn "jax.numpy.fft.fftn")(a\[, s, axes, norm\]) | Compute a multidimensional discrete Fourier transform along given axes. |
| [`fftshift`](_autosummary/jax.numpy.fft.fftshift.html#jax.numpy.fft.fftshift "jax.numpy.fft.fftshift")(x\[, axes\]) | Shift zero-frequency fft component to the center of the spectrum. |
| [`hfft`](_autosummary/jax.numpy.fft.hfft.html#jax.numpy.fft.hfft "jax.numpy.fft.hfft")(a\[, n, axis, norm\]) | Compute a 1-D FFT of an array whose spectrum has Hermitian symmetry. |
| [`ifft`](_autosummary/jax.numpy.fft.ifft.html#jax.numpy.fft.ifft "jax.numpy.fft.ifft")(a\[, n, axis, norm\]) | Compute a one-dimensional inverse discrete Fourier transform. |
| [`ifft2`](_autosummary/jax.numpy.fft.ifft2.html#jax.numpy.fft.ifft2 "jax.numpy.fft.ifft2")(a\[, s, axes, norm\]) | Compute a two-dimensional inverse discrete Fourier transform. |
| [`ifftn`](_autosummary/jax.numpy.fft.ifftn.html#jax.numpy.fft.ifftn "jax.numpy.fft.ifftn")(a\[, s, axes, norm\]) | Compute a multidimensional inverse discrete Fourier transform. |
| [`ifftshift`](_autosummary/jax.numpy.fft.ifftshift.html#jax.numpy.fft.ifftshift "jax.numpy.fft.ifftshift")(x\[, axes\]) | The inverse of [`jax.numpy.fft.fftshift()`](_autosummary/jax.numpy.fft.fftshift.html#jax.numpy.fft.fftshift "jax.numpy.fft.fftshift"). |
| [`ihfft`](_autosummary/jax.numpy.fft.ihfft.html#jax.numpy.fft.ihfft "jax.numpy.fft.ihfft")(a\[, n, axis, norm\]) | Compute a 1-D inverse FFT of an array whose spectrum has Hermitian-symmetry. |
| [`irfft`](_autosummary/jax.numpy.fft.irfft.html#jax.numpy.fft.irfft "jax.numpy.fft.irfft")(a\[, n, axis, norm\]) | Compute a real-valued one-dimensional inverse discrete Fourier transform. |
| [`irfft2`](_autosummary/jax.numpy.fft.irfft2.html#jax.numpy.fft.irfft2 "jax.numpy.fft.irfft2")(a\[, s, axes, norm\]) | Compute a real-valued two-dimensional inverse discrete Fourier transform. |
| [`irfftn`](_autosummary/jax.numpy.fft.irfftn.html#jax.numpy.fft.irfftn "jax.numpy.fft.irfftn")(a\[, s, axes, norm\]) | Compute a real-valued multidimensional inverse discrete Fourier transform. |
| [`rfft`](_autosummary/jax.numpy.fft.rfft.html#jax.numpy.fft.rfft "jax.numpy.fft.rfft")(a\[, n, axis, norm\]) | Compute a one-dimensional discrete Fourier transform of a real-valued array. |
| [`rfft2`](_autosummary/jax.numpy.fft.rfft2.html#jax.numpy.fft.rfft2 "jax.numpy.fft.rfft2")(a\[, s, axes, norm\]) | Compute a two-dimensional discrete Fourier transform of a real-valued array. |
| [`rfftfreq`](_autosummary/jax.numpy.fft.rfftfreq.html#jax.numpy.fft.rfftfreq "jax.numpy.fft.rfftfreq")(n\[, d, dtype, device\]) | Return sample frequencies for the discrete Fourier transform. |
| [`rfftn`](_autosummary/jax.numpy.fft.rfftn.html#jax.numpy.fft.rfftn "jax.numpy.fft.rfftn")(a\[, s, axes, norm\]) | Compute a multidimensional discrete Fourier transform of a real-valued array. |

## jax.numpy.linalg[\#](#module-jax.numpy.linalg "Link to this heading")

|  |  |
|----|----|
| [`cholesky`](_autosummary/jax.numpy.linalg.cholesky.html#jax.numpy.linalg.cholesky "jax.numpy.linalg.cholesky")(a, \*\[, upper, symmetrize_input\]) | Compute the Cholesky decomposition of a matrix. |
| [`cond`](_autosummary/jax.numpy.linalg.cond.html#jax.numpy.linalg.cond "jax.numpy.linalg.cond")(x\[, p\]) | Compute the condition number of a matrix. |
| [`cross`](_autosummary/jax.numpy.linalg.cross.html#jax.numpy.linalg.cross "jax.numpy.linalg.cross")(x1, x2, /, \*\[, axis\]) | Compute the cross-product of two 3D vectors |
| [`det`](_autosummary/jax.numpy.linalg.det.html#jax.numpy.linalg.det "jax.numpy.linalg.det")(a) | Compute the determinant of an array. |
| [`diagonal`](_autosummary/jax.numpy.linalg.diagonal.html#jax.numpy.linalg.diagonal "jax.numpy.linalg.diagonal")(x, /, \*\[, offset\]) | Extract the diagonal of an matrix or stack of matrices. |
| [`eig`](_autosummary/jax.numpy.linalg.eig.html#jax.numpy.linalg.eig "jax.numpy.linalg.eig")(a) | Compute the eigenvalues and eigenvectors of a square array. |
| [`eigh`](_autosummary/jax.numpy.linalg.eigh.html#jax.numpy.linalg.eigh "jax.numpy.linalg.eigh")(a\[, UPLO, symmetrize_input\]) | Compute the eigenvalues and eigenvectors of a Hermitian matrix. |
| [`eigvals`](_autosummary/jax.numpy.linalg.eigvals.html#jax.numpy.linalg.eigvals "jax.numpy.linalg.eigvals")(a) | Compute the eigenvalues of a general matrix. |
| [`eigvalsh`](_autosummary/jax.numpy.linalg.eigvalsh.html#jax.numpy.linalg.eigvalsh "jax.numpy.linalg.eigvalsh")(a\[, UPLO, symmetrize_input\]) | Compute the eigenvalues of a Hermitian matrix. |
| [`inv`](_autosummary/jax.numpy.linalg.inv.html#jax.numpy.linalg.inv "jax.numpy.linalg.inv")(a) | Return the inverse of a square matrix |
| [`lstsq`](_autosummary/jax.numpy.linalg.lstsq.html#jax.numpy.linalg.lstsq "jax.numpy.linalg.lstsq")(a, b\[, rcond, numpy_resid\]) | Return the least-squares solution to a linear equation. |
| [`matmul`](_autosummary/jax.numpy.linalg.matmul.html#jax.numpy.linalg.matmul "jax.numpy.linalg.matmul")(x1, x2, /, \*\[, precision, ...\]) | Perform a matrix multiplication. |
| [`matrix_norm`](_autosummary/jax.numpy.linalg.matrix_norm.html#jax.numpy.linalg.matrix_norm "jax.numpy.linalg.matrix_norm")(x, /, \*\[, keepdims, ord\]) | Compute the norm of a matrix or stack of matrices. |
| [`matrix_power`](_autosummary/jax.numpy.linalg.matrix_power.html#jax.numpy.linalg.matrix_power "jax.numpy.linalg.matrix_power")(a, n) | Raise a square matrix to an integer power. |
| [`matrix_rank`](_autosummary/jax.numpy.linalg.matrix_rank.html#jax.numpy.linalg.matrix_rank "jax.numpy.linalg.matrix_rank")(M\[, rtol, hermitian, tol\]) | Compute the rank of a matrix. |
| [`matrix_transpose`](_autosummary/jax.numpy.linalg.matrix_transpose.html#jax.numpy.linalg.matrix_transpose "jax.numpy.linalg.matrix_transpose")(x, /) | Transpose a matrix or stack of matrices. |
| [`multi_dot`](_autosummary/jax.numpy.linalg.multi_dot.html#jax.numpy.linalg.multi_dot "jax.numpy.linalg.multi_dot")(arrays, \*\[, precision\]) | Efficiently compute matrix products between a sequence of arrays. |
| [`norm`](_autosummary/jax.numpy.linalg.norm.html#jax.numpy.linalg.norm "jax.numpy.linalg.norm")(x\[, ord, axis, keepdims\]) | Compute the norm of a matrix or vector. |
| [`outer`](_autosummary/jax.numpy.linalg.outer.html#jax.numpy.linalg.outer "jax.numpy.linalg.outer")(x1, x2, /) | Compute the outer product of two 1-dimensional arrays. |
| [`pinv`](_autosummary/jax.numpy.linalg.pinv.html#jax.numpy.linalg.pinv "jax.numpy.linalg.pinv")(a\[, rtol, hermitian, rcond\]) | Compute the (Moore-Penrose) pseudo-inverse of a matrix. |
| [`qr`](_autosummary/jax.numpy.linalg.qr.html#jax.numpy.linalg.qr "jax.numpy.linalg.qr")(a\[, mode\]) | Compute the QR decomposition of an array |
| [`slogdet`](_autosummary/jax.numpy.linalg.slogdet.html#jax.numpy.linalg.slogdet "jax.numpy.linalg.slogdet")(a, \*\[, method\]) | Compute the sign and (natural) logarithm of the determinant of an array. |
| [`solve`](_autosummary/jax.numpy.linalg.solve.html#jax.numpy.linalg.solve "jax.numpy.linalg.solve")(a, b) | Solve a linear system of equations. |
| [`svd`](_autosummary/jax.numpy.linalg.svd.html#jax.numpy.linalg.svd "jax.numpy.linalg.svd")(a\[, full_matrices, compute_uv, ...\]) | Compute the singular value decomposition. |
| [`svdvals`](_autosummary/jax.numpy.linalg.svdvals.html#jax.numpy.linalg.svdvals "jax.numpy.linalg.svdvals")(x, /) | Compute the singular values of a matrix. |
| [`tensordot`](_autosummary/jax.numpy.linalg.tensordot.html#jax.numpy.linalg.tensordot "jax.numpy.linalg.tensordot")(x1, x2, /, \*\[, axes, precision, ...\]) | Compute the tensor dot product of two N-dimensional arrays. |
| [`tensorinv`](_autosummary/jax.numpy.linalg.tensorinv.html#jax.numpy.linalg.tensorinv "jax.numpy.linalg.tensorinv")(a\[, ind\]) | Compute the tensor inverse of an array. |
| [`tensorsolve`](_autosummary/jax.numpy.linalg.tensorsolve.html#jax.numpy.linalg.tensorsolve "jax.numpy.linalg.tensorsolve")(a, b\[, axes\]) | Solve the tensor equation a x = b for x. |
| [`trace`](_autosummary/jax.numpy.linalg.trace.html#jax.numpy.linalg.trace "jax.numpy.linalg.trace")(x, /, \*\[, offset, dtype\]) | Compute the trace of a matrix. |
| [`vector_norm`](_autosummary/jax.numpy.linalg.vector_norm.html#jax.numpy.linalg.vector_norm "jax.numpy.linalg.vector_norm")(x, /, \*\[, axis, keepdims, ord\]) | Compute the vector norm of a vector or batch of vectors. |
| [`vecdot`](_autosummary/jax.numpy.linalg.vecdot.html#jax.numpy.linalg.vecdot "jax.numpy.linalg.vecdot")(x1, x2, /, \*\[, axis, precision, ...\]) | Compute the (batched) vector conjugate dot product of two arrays. |

## JAX Array[\#](#jax-array "Link to this heading")

The JAX [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") (along with its alias, [`jax.numpy.ndarray`](_autosummary/jax.numpy.ndarray.html#jax.numpy.ndarray "jax.numpy.ndarray")) is the core array object in JAX: you can think of it as JAX’s equivalent of a [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)"). Like [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)"), most users will not need to instantiate [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") objects manually, but rather will create them via [`jax.numpy`](#module-jax.numpy "jax.numpy") functions like [`array()`](_autosummary/jax.numpy.array.html#jax.numpy.array "jax.numpy.array"), [`arange()`](_autosummary/jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange"), [`linspace()`](_autosummary/jax.numpy.linspace.html#jax.numpy.linspace "jax.numpy.linspace"), and others listed above.

### Copying and Serialization[\#](#copying-and-serialization "Link to this heading")

JAX [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") objects are designed to work seamlessly with Python standard library tools where appropriate.

With the built-in [`copy`](https://docs.python.org/3/library/copy.html#module-copy "(in Python v3.14)") module, when [`copy.copy()`](https://docs.python.org/3/library/copy.html#copy.copy "(in Python v3.14)") or [`copy.deepcopy()`](https://docs.python.org/3/library/copy.html#copy.deepcopy "(in Python v3.14)") encounder an [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array"), it is equivalent to calling the [`copy()`](_autosummary/jax.Array.copy.html#jax.Array.copy "jax.Array.copy") method, which will create a copy of the buffer on the same device as the original array. This will work correctly within traced/JIT-compiled code, though copy operations may be elided by the compiler in this context.

When the built-in [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle "(in Python v3.14)") module encounters an [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array"), it will be serialized via a compact bit representation in a similar manner to pickled [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray "(in NumPy v2.4)") objects. When unpickled, the result will be a new [`Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") object *on the default device.* This is because in general, pickling and unpickling may take place in different runtime environments, and there is no general way to map the device IDs of one runtime to the device IDs of another. If [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle "(in Python v3.14)") is used in traced/JIT-compiled code, it will result in a [`ConcretizationTypeError`](errors.html#jax.errors.ConcretizationTypeError "jax.errors.ConcretizationTypeError").

## Python Array API standard[\#](#python-array-api-standard "Link to this heading")

Note

Prior to JAX v0.4.32, you must `import`` ``jax.experimental.array_api` in order to enable the array API for JAX arrays. After JAX v0.4.32, importing this module is no longer required, and will raise a deprecation warning. After JAX v0.5.0, this import will raise an error.

Starting with JAX v0.4.32, [`jax.Array`](_autosummary/jax.Array.html#jax.Array "jax.Array") and [`jax.numpy`](#module-jax.numpy "jax.numpy") are compatible with the [Python Array API Standard](https://data-apis.org/array-api). You can access the Array API namespace via `jax.Array.__array_namespace__()`:

    >>> def f(x):
    ...   nx = x.__array_namespace__()
    ...   return nx.sin(x) ** 2 + nx.cos(x) ** 2

    >>> import jax.numpy as jnp
    >>> x = jnp.arange(5)
    >>> f(x).round()
    Array([1., 1., 1., 1., 1.], dtype=float32)

JAX departs from the standard in a few places, namely because JAX arrays are immutable, in-place updates are not supported. Some of these incompatibilities are being addressed via the [array-api-compat](https://github.com/data-apis/array-api-compat) module.

For more information, refer to the [Python Array API Standard](https://data-apis.org/array-api) documentation.

[](jax.html "previous page")

previous

API Reference

[](_autosummary/jax.numpy.ndarray.at.html "next page")

next

jax.numpy.ndarray.at

Contents

- [jax.numpy.fft](#module-jax.numpy.fft)
- [jax.numpy.linalg](#module-jax.numpy.linalg)
- [JAX Array](#jax-array)
  - [Copying and Serialization](#copying-and-serialization)
- [Python Array API standard](#python-array-api-standard)

By The JAX authors

© Copyright 2024, The JAX Authors.\
