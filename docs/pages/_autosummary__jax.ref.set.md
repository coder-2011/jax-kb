- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.set

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.set.rst "Download source file")
-  .pdf

# jax.ref.set

## Contents

- [`set()`](#jax.ref.set)

# jax.ref.set[\#](#jax-ref-set "Link to this heading")

jax.ref.set(*ref*, *idx*, *value*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/state/primitives.py#L277-L322)[\#](#jax.ref.set "Link to this definition")  
Set a value in an Ref in-place.

This is equivalent to `ref[idx]`` ``=`` ``value` for a NumPy-style indexer `idx`. For more on mutable array refs, refer to the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html).

Parameters:  
- **ref** ([*core.Ref*](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref") *\|* *TransformedRef*) – a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref") object. On return, the buffer will be mutated by this operation.

- **idx** (*Indexer* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Indexer,* *...\]* *\|* *None*) – a NumPy-style indexer

- **value** (*ArrayLike* *\|* *HijaxType*) – a [`jax.Array`](jax.Array.html#jax.Array "jax.Array") object (note, not a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) containing the values to set in the array.

Returns:  
None

Return type:  
None

Examples

    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.zeros(5))
    >>> jax.ref.set(ref, 1, 10.0)
    >>> ref
    Ref([ 0., 10.,  0.,  0.,  0.], dtype=float32)

Equivalent operation via indexing syntax:

    >>> ref = jax.new_ref(jax.numpy.zeros(5))
    >>> ref[1] = 10.0
    >>> ref
    Ref([ 0., 10.,  0.,  0.,  0.], dtype=float32)

Use `...` to set the value of a scalar ref:

    >>> ref = jax.new_ref(jax.numpy.int32(0))
    >>> ref[...] = 4
    >>> ref
    Ref(4, dtype=int32)

[](jax.ref.new_ref.html "previous page")

previous

jax.ref.new_ref

[](jax.ref.swap.html "next page")

next

jax.ref.swap

Contents

- [`set()`](#jax.ref.set)

By The JAX authors

© Copyright 2024, The JAX Authors.\
