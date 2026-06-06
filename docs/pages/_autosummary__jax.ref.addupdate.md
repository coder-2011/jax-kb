- [](../index.html)
- [API Reference](../jax.html)
- [`jax.ref` module](../jax.ref.html)
- jax.ref.addupdate

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/_autosummary/jax.ref.addupdate.rst "Download source file")
-  .pdf

# jax.ref.addupdate

## Contents

- [`addupdate()`](#jax.ref.addupdate)

# jax.ref.addupdate[\#](#jax-ref-addupdate "Link to this heading")

jax.ref.addupdate(*ref*, *idx*, *x*)[\[source\]](https://github.com/jax-ml/jax/blob/main/jax/_src/state/primitives.py#L341-L390)[\#](#jax.ref.addupdate "Link to this definition")  
Add to an element in an Ref in-place.

This is analogous to `ref[idx]`` ``+=`` ``value` for a NumPy array `ref` and NumPy-style indexer `idx`. However, for an Ref `ref`, executing `ref[idx]`` ``+=`` ``value` actually performs a `ref_get`, add, and `ref_set`, so using this function can be more efficient under autodiff. For more on mutable array refs, refer to the [Ref guide](https://docs.jax.dev/en/latest/array_refs.html).

Parameters:  
- **ref** ([*core.Ref*](jax.ref.Ref.html#jax.ref.Ref "jax._src.core.Ref") *\|* *TransformedRef*) – a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref") object. On return, the buffer will be mutated by this operation.

- **idx** (*Indexer* *\|* [*tuple*](https://docs.python.org/3/library/stdtypes.html#tuple "(in Python v3.14)")*\[Indexer,* *...\]* *\|* *None*) – a NumPy-style indexer

- **x** (*ArrayLike* *\|* *HijaxType*) – a [`jax.Array`](jax.Array.html#jax.Array "jax.Array") object (note, not a [`jax.ref.Ref`](jax.ref.Ref.html#jax.ref.Ref "jax.ref.Ref")) containing the values to add at the specified indices.

Returns:  
None

Return type:  
None

Examples

    >>> import jax
    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> jax.ref.addupdate(ref, 2, 10)
    >>> ref
    Ref([ 0,  1, 12,  3,  4], dtype=int32)

Equivalent operation via indexing syntax:

    >>> ref = jax.new_ref(jax.numpy.arange(5))
    >>> ref[2] += 10
    >>> ref
    Ref([ 0,  1, 12,  3,  4], dtype=int32)

Use `...` to add to a scalar ref:

    >>> ref = jax.new_ref(jax.numpy.int32(2))
    >>> ref[...] += 10
    >>> ref
    Ref(12, dtype=int32)

[](jax.ref.swap.html "previous page")

previous

jax.ref.swap

[](../jax.stages.html "next page")

next

`jax.stages` module

Contents

- [`addupdate()`](#jax.ref.addupdate)

By The JAX authors

© Copyright 2024, The JAX Authors.\
