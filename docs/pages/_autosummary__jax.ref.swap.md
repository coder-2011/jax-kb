- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.swap

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.swap.rst "Download source file")
-  .pdf

# jax.ref.swap

## Contents

- [`swap()`](#jax.ref.swap)

# jax.ref.swap[\#](#jax-ref-swap "Link to this heading")

jax.ref.swap(*ref*, *idx*, *value*, *\_function_name='ref_swap'*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/state/primitives.py#L201-L260)[\#](#jax.ref.swap "Link to this definition")  
Update an array value inplace while returning the previous value.

This is equivalent to `ref[idx],`` ``prev`` ``=`` ``value,`` ``ref[idx]` while returning `prev`, for a NumPy-style indexer `idx`. For more on mutable array refs, refer to the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html).

Parameters:  
- **ref** ([*core.Ref*](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref") *\|* *TransformedRef*) – a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref") object. On return, the buffer will be mutated by this operation.

- **idx** (*Indexer* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Indexer,* *...\]* *\|* *None*) – a NumPy-style indexer

- **value** (*ArrayLike* *\|* *HijaxType*) – a [`jax.Array`](jax.Array.html#jax.Array "jax.Array") object (note, not a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) containing the values to set in the array.

- **\_function_name** ([*str*](https://docs.python.org/3/library/stdtypes.html#str "(in Python v3.14)"))

Returns:  
A [`jax.Array`](jax.Array.html#jax.Array "jax.Array") containing the previous value at idx.

Return type:  
[Array](jax.Array.html#jax.Array "jax.Array") \| HijaxType

Examples

    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> jax.ref.swap(ref, 3, 10)
    Array(3, dtype=int32)
    >>> ref
    Ref([ 0,  1,  2, 10,  4], dtype=int32)

Equivalent operation via indexing syntax:

    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> ref[3], prev = 10, ref[3]
    >>> prev
    Array(3, dtype=int32)
    >>> ref
    Ref([ 0,  1,  2, 10,  4], dtype=int32)

Use `...` to swap the value of a scalar ref:

    >>> ref = jax.new_ref(jax.numpy.int32(5))
    >>> jax.ref.swap(ref, ..., 10)
    Array(5, dtype=int32)
    >>> ref
    Ref(10, dtype=int32)

[](jax.ref.set.html "previous page")

previous

jax.ref.set

[](jax.ref.addupdate.html "next page")

next

jax.ref.addupdate

Contents

- [`swap()`](#jax.ref.swap)

By The JAX authors

© Copyright 2024, The JAX Authors.\
