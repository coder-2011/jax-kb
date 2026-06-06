- [](../index.html)
- [API Reference](../jax.html)
- jax.Array.at

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.Array.at.rst "Download source file")
-  .pdf

# jax.Array.at

## Contents

- [`Array.at`](#jax.Array.at)

# jax.Array.at[\#](#jax-array-at "Link to this heading")

*abstract property* Array.at[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/numpy/array_methods.py#L1036-L1148)[\#](#jax.Array.at "Link to this definition")  
Helper property for index update functionality.

The `at` property provides a functionally pure equivalent of in-place array modifications.

In particular:

| Alternate syntax | Equivalent In-place expression |
|----|----|
| `x`` ``=`` ``x.at[idx].set(y)` | `x[idx]`` ``=`` ``y` |
| `x`` ``=`` ``x.at[idx].add(y)` | `x[idx]`` ``+=`` ``y` |
| `x`` ``=`` ``x.at[idx].subtract(y)` | `x[idx]`` ``-=`` ``y` |
| `x`` ``=`` ``x.at[idx].multiply(y)` | `x[idx]`` ``*=`` ``y` |
| `x`` ``=`` ``x.at[idx].divide(y)` | `x[idx]`` ``/=`` ``y` |
| `x`` ``=`` ``x.at[idx].power(y)` | `x[idx]`` ``**=`` ``y` |
| `x`` ``=`` ``x.at[idx].min(y)` | `x[idx]`` ``=`` ``minimum(x[idx],`` ``y)` |
| `x`` ``=`` ``x.at[idx].max(y)` | `x[idx]`` ``=`` ``maximum(x[idx],`` ``y)` |
| `x`` ``=`` ``x.at[idx].apply(ufunc)` | `ufunc.at(x,`` ``idx)` |
| `x`` ``=`` ``x.at[idx].get()` | `x`` ``=`` ``x[idx]` |

None of the `x.at` expressions modify the original `x`; instead they return a modified copy of `x`. However, inside a [`jit()`](jax.jit.html#jax.jit "jax.jit") compiled function, expressions like `x`` ``=`` ``x.at[idx].set(y)` are guaranteed to be applied in-place.

Unlike NumPy in-place operations such as `x[idx]`` ``+=`` ``y`, if multiple indices refer to the same location, all updates will be applied (NumPy would only apply the last update, rather than applying all updates.) The order in which conflicting updates are applied is implementation-defined and may be nondeterministic (e.g., due to concurrency on some hardware platforms).

By default, JAX assumes that all indices are in-bounds. Alternative out-of-bound index semantics can be specified via the `mode` parameter (see below).

Parameters:  
- **mode** –

  string specifying out-of-bound indexing mode. Options are:

  - `"promise_in_bounds"`: (default) The user promises that indices are in bounds. No additional checking will be performed. In practice, this means that out-of-bounds indices in `get()` will be clipped, and out-of-bounds indices in `set()`, `add()`, etc. will be dropped.

  - `"clip"`: clamp out of bounds indices into valid range.

  - `"drop"`: ignore out-of-bound indices.

  - `"fill"`: alias for `"drop"`. For get(), the optional `fill_value` argument specifies the value that will be returned.

  See [`jax.lax.GatherScatterMode`](../jax.lax.html#jax.lax.GatherScatterMode "jax.lax.GatherScatterMode") for more details.

- **wrap_negative_indices** – If True (default) then negative indices indicate position from the end of the array, similar to Python and NumPy indexing. If False, then negative indices are considered out-of-bounds and behave according to the `mode` parameter.

- **fill_value** – Only applies to the `get()` method: the fill value to return for out-of-bounds slices when `mode` is `'fill'`. Ignored otherwise. Defaults to `NaN` for inexact types, the largest negative value for signed types, the largest positive value for unsigned types, and `True` for booleans.

- **indices_are_sorted** – If True, the implementation will assume that the (normalized) indices passed to `at[]` are sorted in ascending order, which can lead to more efficient execution on some backends. If True but the indices are not actually sorted, the output is undefined.

- **unique_indices** – If True, the implementation will assume that the (normalized) indices passed to `at[]` are unique, which can result in more efficient execution on some backends. If True but the indices are not actually unique, the output is undefined.

Examples

    >>> x = jnp.arange(5.0)
    >>> x
    Array([0., 1., 2., 3., 4.], dtype=float32)
    >>> x.at[2].get()
    Array(2., dtype=float32)
    >>> x.at[2].add(10)
    Array([ 0.,  1., 12.,  3.,  4.], dtype=float32)

By default, out-of-bound indices are ignored in updates, but this behavior can be controlled with the `mode` parameter:

    >>> x.at[10].add(10)  # dropped
    Array([0., 1., 2., 3., 4.], dtype=float32)
    >>> x.at[20].add(10, mode='clip')  # clipped
    Array([ 0.,  1.,  2.,  3., 14.], dtype=float32)

For `get()`, out-of-bound indices are clipped by default:

    >>> x.at[20].get()  # out-of-bounds indices clipped
    Array(4., dtype=float32)
    >>> x.at[20].get(mode='fill')  # out-of-bounds indices filled with NaN
    Array(nan, dtype=float32)
    >>> x.at[20].get(mode='fill', fill_value=-1)  # custom fill value
    Array(-1., dtype=float32)

Negative indices count from the end of the array, but this behavior can be disabled by setting `wrap_negative_indices`` ``=`` ``False`:

    >>> x.at[-1].set(99)
    Array([ 0.,  1.,  2.,  3., 99.], dtype=float32)
    >>> x.at[-1].set(99, wrap_negative_indices=False, mode='drop')  # dropped!
    Array([0., 1., 2., 3., 4.], dtype=float32)

[](jax.Array.astype.html "previous page")

previous

jax.Array.astype

[](jax.Array.byteswap.html "next page")

next

jax.Array.byteswap

Contents

- [`Array.at`](#jax.Array.at)

By The JAX authors

© Copyright 2024, The JAX Authors.\
