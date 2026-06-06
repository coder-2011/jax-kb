- [](../index.html)
- [API Reference](../jax.html)
- jax.Array

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.rst "Download source file")
-  .pdf

# jax.Array

## Contents

- [`Array`](#jax.Array)
  - [`Array.__init__()`](#jax.Array.__init__)

# jax.Array[\#](#jax-array "Link to this heading")

*class* jax.Array[\#](#jax.Array "Link to this definition")  
Array base class for JAX

`jax.Array` is the public interface for instance checks and type annotation of JAX arrays and tracers. Its main applications are in instance checks and type annotations; for example:

    x = jnp.arange(5)
    isinstance(x, jax.Array)  # returns True both inside and outside traced functions.

    def f(x: Array) -> Array:  # type annotations are valid for traced and non-traced types.
      return x

`jax.Array` should not be used directly for creation of arrays; instead you should use array creation routines offered in [`jax.numpy`](../jax.numpy.html#module-jax.numpy "jax.numpy"), such as [`jax.numpy.array()`](jax.numpy.array.html#jax.numpy.array "jax.numpy.array"), [`jax.numpy.zeros()`](jax.numpy.zeros.html#jax.numpy.zeros "jax.numpy.zeros"), [`jax.numpy.ones()`](jax.numpy.ones.html#jax.numpy.ones "jax.numpy.ones"), [`jax.numpy.full()`](jax.numpy.full.html#jax.numpy.full "jax.numpy.full"), [`jax.numpy.arange()`](jax.numpy.arange.html#jax.numpy.arange "jax.numpy.arange"), etc.

\_\_init\_\_()[\#](#jax.Array.__init__ "Link to this definition")  

Methods

|  |  |
|----|----|
| [`__init__`](#jax.Array.__init__ "jax.Array.__init__")() |  |
| `addressable_data`(index) | Return an array of the addressable data at a particular index. |
| [`all`](jax.Array.all.html#jax.Array.all "jax.Array.all")(\[axis, out, keepdims, where\]) | Test whether all array elements along a given axis evaluate to True. |
| [`any`](jax.Array.any.html#jax.Array.any "jax.Array.any")(\[axis, out, keepdims, where\]) | Test whether any array elements along a given axis evaluate to True. |
| [`argmax`](jax.Array.argmax.html#jax.Array.argmax "jax.Array.argmax")(\[axis, out, keepdims\]) | Return the index of the maximum value. |
| [`argmin`](jax.Array.argmin.html#jax.Array.argmin "jax.Array.argmin")(\[axis, out, keepdims\]) | Return the index of the minimum value. |
| [`argpartition`](jax.Array.argpartition.html#jax.Array.argpartition "jax.Array.argpartition")(kth\[, axis\]) | Return the indices that partially sort the array. |
| [`argsort`](jax.Array.argsort.html#jax.Array.argsort "jax.Array.argsort")(\[axis, kind, order, stable, descending\]) | Return the indices that sort the array. |
| [`astype`](jax.Array.astype.html#jax.Array.astype "jax.Array.astype")(dtype\[, copy, device\]) | Copy the array and cast to a specified dtype. |
| [`byteswap`](jax.Array.byteswap.html#jax.Array.byteswap "jax.Array.byteswap")() | Swap the bytes of the array elements. |
| [`choose`](jax.Array.choose.html#jax.Array.choose "jax.Array.choose")(choices\[, out, mode\]) | Construct an array choosing from elements of multiple arrays. |
| [`clip`](jax.Array.clip.html#jax.Array.clip "jax.Array.clip")(\[min, max\]) | Return an array whose values are limited to a specified range. |
| [`compress`](jax.Array.compress.html#jax.Array.compress "jax.Array.compress")(condition\[, axis, out, size, ...\]) | Return selected slices of this array along given axis. |
| [`conj`](jax.Array.conj.html#jax.Array.conj "jax.Array.conj")() | Return the complex conjugate of the array. |
| [`conjugate`](jax.Array.conjugate.html#jax.Array.conjugate "jax.Array.conjugate")() | Return the complex conjugate of the array. |
| [`copy`](jax.Array.copy.html#jax.Array.copy "jax.Array.copy")() | Return a copy of the array. |
| [`copy_to_host_async`](jax.Array.copy_to_host_async.html#jax.Array.copy_to_host_async "jax.Array.copy_to_host_async")() | Copies an `Array` to the host asynchronously. |
| [`cumprod`](jax.Array.cumprod.html#jax.Array.cumprod "jax.Array.cumprod")(\[axis, dtype, out\]) | Return the cumulative product of the array. |
| [`cumsum`](jax.Array.cumsum.html#jax.Array.cumsum "jax.Array.cumsum")(\[axis, dtype, out\]) | Return the cumulative sum of the array. |
| [`diagonal`](jax.Array.diagonal.html#jax.Array.diagonal "jax.Array.diagonal")(\[offset, axis1, axis2\]) | Return the specified diagonal from the array. |
| [`dot`](jax.Array.dot.html#jax.Array.dot "jax.Array.dot")(b, \*\[, precision, preferred_element_type\]) | Compute the dot product of two arrays. |
| [`flatten`](jax.Array.flatten.html#jax.Array.flatten "jax.Array.flatten")(\[order, out_sharding\]) | Flatten array into a 1-dimensional shape. |
| [`item`](jax.Array.item.html#jax.Array.item "jax.Array.item")(\*args) | Copy an element of an array to a standard Python scalar and return it. |
| [`max`](jax.Array.max.html#jax.Array.max "jax.Array.max")(\[axis, out, keepdims, initial, where\]) | Return the maximum of array elements along a given axis. |
| [`mean`](jax.Array.mean.html#jax.Array.mean "jax.Array.mean")(\[axis, dtype, out, keepdims, where\]) | Return the mean of array elements along a given axis. |
| [`min`](jax.Array.min.html#jax.Array.min "jax.Array.min")(\[axis, out, keepdims, initial, where\]) | Return the minimum of array elements along a given axis. |
| [`nonzero`](jax.Array.nonzero.html#jax.Array.nonzero "jax.Array.nonzero")(\*\[, fill_value, size\]) | Return indices of nonzero elements of an array. |
| [`prod`](jax.Array.prod.html#jax.Array.prod "jax.Array.prod")(\[axis, dtype, out, keepdims, initial, ...\]) | Return product of the array elements over a given axis. |
| [`ptp`](jax.Array.ptp.html#jax.Array.ptp "jax.Array.ptp")(\[axis, out, keepdims\]) | Return the peak-to-peak range along a given axis. |
| [`ravel`](jax.Array.ravel.html#jax.Array.ravel "jax.Array.ravel")(\[order, out_sharding\]) | Flatten array into a 1-dimensional shape. |
| [`repeat`](jax.Array.repeat.html#jax.Array.repeat "jax.Array.repeat")(repeats\[, axis, total_repeat_length, ...\]) | Construct an array from repeated elements. |
| [`reshape`](jax.Array.reshape.html#jax.Array.reshape "jax.Array.reshape")(\*args\[, order, out_sharding\]) | Returns an array containing the same data with a new shape. |
| [`round`](jax.Array.round.html#jax.Array.round "jax.Array.round")(\[decimals, out\]) | Round array elements to a given decimal. |
| [`searchsorted`](jax.Array.searchsorted.html#jax.Array.searchsorted "jax.Array.searchsorted")(v\[, side, sorter, method\]) | Perform a binary search within a sorted array. |
| [`sort`](jax.Array.sort.html#jax.Array.sort "jax.Array.sort")(\[axis, kind, order, stable, descending\]) | Return a sorted copy of an array. |
| [`squeeze`](jax.Array.squeeze.html#jax.Array.squeeze "jax.Array.squeeze")(\[axis\]) | Remove one or more length-1 axes from array. |
| [`std`](jax.Array.std.html#jax.Array.std "jax.Array.std")(\[axis, dtype, out, ddof, keepdims, ...\]) | Compute the standard deviation along a given axis. |
| [`sum`](jax.Array.sum.html#jax.Array.sum "jax.Array.sum")(\[axis, dtype, out, keepdims, initial, ...\]) | Sum of the elements of the array over a given axis. |
| [`swapaxes`](jax.Array.swapaxes.html#jax.Array.swapaxes "jax.Array.swapaxes")(axis1, axis2) | Swap two axes of an array. |
| [`take`](jax.Array.take.html#jax.Array.take "jax.Array.take")(indices\[, axis, out, mode, ...\]) | Take elements from an array. |
| [`to_device`](jax.Array.to_device.html#jax.Array.to_device "jax.Array.to_device")(device, \*\[, stream\]) | Return a copy of the array on the specified device |
| [`trace`](jax.Array.trace.html#jax.Array.trace "jax.Array.trace")(\[offset, axis1, axis2, dtype, out\]) | Return the sum along the diagonal. |
| [`transpose`](jax.Array.transpose.html#jax.Array.transpose "jax.Array.transpose")(\*args) | Returns a copy of the array with axes transposed. |
| [`var`](jax.Array.var.html#jax.Array.var "jax.Array.var")(\[axis, dtype, out, ddof, keepdims, ...\]) | Compute the variance along a given axis. |
| [`view`](jax.Array.view.html#jax.Array.view "jax.Array.view")(\[dtype, type\]) | Return a bitwise copy of the array, viewed as a new dtype. |

Attributes

|  |  |
|----|----|
| [`T`](jax.Array.T.html#jax.Array.T "jax.Array.T") | Compute the all-axis array transpose. |
| [`addressable_shards`](jax.Array.addressable_shards.html#jax.Array.addressable_shards "jax.Array.addressable_shards") | List of addressable shards. |
| [`at`](jax.Array.at.html#jax.Array.at "jax.Array.at") | Helper property for index update functionality. |
| [`committed`](jax.Array.committed.html#jax.Array.committed "jax.Array.committed") | Whether the array is committed or not. |
| [`device`](jax.Array.device.html#jax.Array.device "jax.Array.device") | Array API-compatible device attribute. |
| [`dtype`](jax.Array.dtype.html#jax.Array.dtype "jax.Array.dtype") | The data type ([`numpy.dtype`](jax.numpy.dtype.html#jax.numpy.dtype "numpy.dtype")) of the array. |
| [`flat`](jax.Array.flat.html#jax.Array.flat "jax.Array.flat") | Use [`flatten()`](jax.Array.flatten.html#jax.Array.flatten "jax.Array.flatten") instead. |
| [`global_shards`](jax.Array.global_shards.html#jax.Array.global_shards "jax.Array.global_shards") | List of global shards. |
| [`imag`](jax.Array.imag.html#jax.Array.imag "jax.Array.imag") | Return the imaginary part of the array. |
| [`is_fully_addressable`](jax.Array.is_fully_addressable.html#jax.Array.is_fully_addressable "jax.Array.is_fully_addressable") | Is this Array fully addressable? |
| [`is_fully_replicated`](jax.Array.is_fully_replicated.html#jax.Array.is_fully_replicated "jax.Array.is_fully_replicated") | Is this Array fully replicated? |
| [`itemsize`](jax.Array.itemsize.html#jax.Array.itemsize "jax.Array.itemsize") | Length of one array element in bytes. |
| [`mT`](jax.Array.mT.html#jax.Array.mT "jax.Array.mT") | Compute the (batched) matrix transpose. |
| [`nbytes`](jax.Array.nbytes.html#jax.Array.nbytes "jax.Array.nbytes") | Total bytes consumed by the elements of the array. |
| [`ndim`](jax.Array.ndim.html#jax.Array.ndim "jax.Array.ndim") | The number of dimensions in the array. |
| [`real`](jax.Array.real.html#jax.Array.real "jax.Array.real") | Return the real part of the array. |
| [`shape`](jax.Array.shape.html#jax.Array.shape "jax.Array.shape") | The shape of the array. |
| [`sharding`](jax.Array.sharding.html#jax.Array.sharding "jax.Array.sharding") | The sharding for the array. |
| [`size`](jax.Array.size.html#jax.Array.size "jax.Array.size") | The total number of elements in the array. |

[](jax.custom_batching.sequential_vmap.html "previous page")

previous

jax.custom_batching.sequential_vmap

[](jax.make_array_from_callback.html "next page")

next

jax.make_array_from_callback

Contents

- [`Array`](#jax.Array)
  - [`Array.__init__()`](#jax.Array.__init__)

By The JAX authors

© Copyright 2024, The JAX Authors.\
